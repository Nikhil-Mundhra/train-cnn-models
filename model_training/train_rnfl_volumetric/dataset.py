"""
dataset.py
==========
PyTorch Dataset for paired Optovue Solix DICOM OCT volumes and ground-truth curves.
Supports:
- 2.5D multi-slice context (e.g. 5 adjacent B-scans: z-2 to z+2)
- Strict subject-level grouping
- Standardized anatomical eye orientation (horizontal flipping for OS)
- Aligned volumetric binary RNFL mask generation
- Continuous surface extraction (ILM and NFL) and optic cup absence flag
"""

import os
import glob
import re
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset

AXIAL_UM = 3.12367
FAST_UM = 18.7500
SLOW_UM = 18.8100
SENTINEL = 3000
DROP = {'N/A', 'RPE Ref'}

_D = re.compile(rb'<D>\s*(-?\d+)\s*</D>')
_TYPE = re.compile(rb'<Type>\s*([^<]+?)\s*</Type>')
_ARRAY = re.compile(rb'<ARRAY>\s*(\d+)\s*</ARRAY>')
_IMG = re.compile(rb'<IMAGE_Number>\s*(\d+)\s*</IMAGE_Number>')


def find_matching_curve_xml(tsv_dir, subj, eye, dcm_fname, protocol="Disc Cube"):
    """
    Finds the exact XML curve corresponding to a DICOM volume.
    Matches via timestamp in the master XML if available, otherwise falls back to glob.
    """
    subj_tsv = os.path.join(tsv_dir, subj)
    if not os.path.exists(subj_tsv):
        return None

    # Try matching by timestamp via master XML
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", dcm_fname)
    if m:
        target_time = f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}"
        master_xmls = glob.glob(os.path.join(subj_tsv, "*.xml"))
        for m_xml in master_xmls:
            try:
                tree = ET.parse(m_xml)
                for scan in tree.getroot().iter("Scan"):
                    stype = scan.findtext("ScanType", "").strip()
                    stime = scan.findtext("ScanTime", "").strip()
                    seye = scan.findtext("Eye", "").strip()
                    cfile = scan.findtext("Curve/File", "").strip()
                    if protocol.lower() in stype.lower() and seye == eye and stime == target_time and cfile:
                        rel_path = cfile.replace("\\", "/").lstrip("./")
                        full_path = os.path.join(subj_tsv, rel_path)
                        if os.path.exists(full_path):
                            return full_path
            except Exception:
                pass

    # Fallback to globbing by eye and protocol if timestamp matching yields no result
    xmls = glob.glob(os.path.join(subj_tsv, "curve", f"*{eye}*{protocol}*.xml"))
    if not xmls:
        xmls = glob.glob(os.path.join(subj_tsv, "curve", f"*{protocol}*{eye}*.xml"))
    return xmls[0] if xmls else None


def find_dicom_pixel_offset(path):
    """
    Finds the byte offset of the PixelData tag (7FE0, 0010) in a DICOM file.
    """
    with open(path, 'rb') as fh:
        head = fh.read(1 << 23)
        tag = head.find(b'\xe0\x7f\x10\x00')
        vr = head[tag + 4:tag + 6]
        return tag + 12 if vr in (b'OW', b'OB', b'UN') else tag + 8


def read_bscan_raw(path, index, rows=768, cols=320, start=None):
    """
    Directly seeks and reads a single 16-bit uint B-scan from DICOM PixelData.
    """
    nbytes = rows * cols * 2
    if start is None:
        start = find_dicom_pixel_offset(path)
    with open(path, 'rb') as fh:
        fh.seek(start + index * nbytes)
        buf = fh.read(nbytes)
    return np.frombuffer(buf, dtype='<u2').reshape(rows, cols)


def load_curves(path, mask_sentinel=True):
    """
    Fast regex parser for curve XML files.
    Returns dictionary of 2D numpy arrays (B-scans x A-scans).
    """
    with open(path, 'rb') as fh:
        blob = fh.read()
    types = [t.decode().strip() for t in _TYPE.findall(blob)]
    widths = {int(a) for a in _ARRAY.findall(blob)}
    n_img = len(_IMG.findall(blob))
    assert len(widths) == 1, f"Ragged array lengths in {path}"
    width = widths.pop()
    per = len(types) // n_img
    order = types[:per]

    cube = np.array(_D.findall(blob), dtype=np.int32).reshape(n_img, per, width)
    out = {}
    for j, name in enumerate(order):
        if name in DROP:
            continue
        a = cube[:, j, :].astype(float)
        if mask_sentinel:
            a[a >= SENTINEL] = np.nan
            a[a <= 0] = np.nan
        out[name] = a
    return out


class SolixRNFLDataset(Dataset):
    """
    PyTorch Dataset loading 2.5D multi-slice B-scans and paired RNFL targets.
    """
    def __init__(
        self,
        dataset_root="/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified",
        subjects=None,
        arm="good",
        protocol="Disc Cube",
        context_slices=5,
        standardize_eye=True,
        transform=None
    ):
        self.dataset_root = dataset_root
        self.arm = arm
        self.protocol = protocol
        self.context_slices = context_slices
        self.half_ctx = context_slices // 2
        self.standardize_eye = standardize_eye
        self.transform = transform

        tsv_dir = os.path.join(dataset_root, "tsv", arm)
        dcm_dir = os.path.join(dataset_root, "dicom")

        all_subjects = sorted([d for d in os.listdir(tsv_dir) if not d.startswith('.')])
        if subjects is not None:
            self.subjects = [s for s in all_subjects if s in subjects]
        else:
            self.subjects = all_subjects

        self.scans = []
        self.samples = []

        # Index all available scans and samples with exact timestamp fidelity
        for subj in self.subjects:
            dcm_matches = sorted(glob.glob(os.path.join(dcm_dir, subj, f"*{protocol}*_OPT.dcm")))
            for dcm_path in dcm_matches:
                dcm_fname = os.path.basename(dcm_path)
                eye = "OS" if "_OS_" in dcm_fname else "OD"

                # Find matching XML curve with timestamp fidelity
                xml_path = find_matching_curve_xml(tsv_dir, subj, eye, dcm_fname, protocol=protocol)
                if not xml_path:
                    continue

                # Load and cache curves for this scan
                curves = load_curves(xml_path, mask_sentinel=True)
                n_bscans, n_ascans = curves['ILM'].shape

                # Compute optic disc center and radius
                nan_mask = np.isnan(curves['NFL'])
                zc, xc = np.mean(np.where(nan_mask)[0]), np.mean(np.where(nan_mask)[1])
                r_disc = np.sqrt(len(np.where(nan_mask)[0]) / np.pi)

                # Find pixel data offset and open read-only memory map
                pixel_offset = find_dicom_pixel_offset(dcm_path)
                memmap = np.memmap(dcm_path, dtype='<u2', mode='r', offset=pixel_offset, shape=(n_bscans, 768, 320))

                scan_info = {
                    'subject': subj,
                    'eye': eye,
                    'xml_path': xml_path,
                    'dcm_path': dcm_path,
                    'curves': curves,
                    'memmap': memmap,
                    'n_bscans': n_bscans,
                    'n_ascans': n_ascans,
                    'zc': zc,
                    'xc': xc,
                    'r_disc': r_disc,
                }
                scan_idx = len(self.scans)
                self.scans.append(scan_info)

                # Add each B-scan slice
                for b_idx in range(n_bscans):
                    # Check if peripapillary slice
                    dist_z = abs(b_idx - zc)
                    is_peripapillary = (dist_z <= 2.0 * r_disc)
                    self.samples.append({
                        'scan_idx': scan_idx,
                        'bscan_idx': b_idx,
                        'is_peripapillary': is_peripapillary
                    })

        print(f"[SolixRNFLDataset] Loaded {len(self.scans)} scans, {len(self.samples)} B-scans across {len(self.subjects)} subjects.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        scan = self.scans[sample['scan_idx']]
        b_idx = sample['bscan_idx']
        n_bscans = scan['n_bscans']

        # 1. Load 2.5D multi-slice stack from zero-overhead memory map
        slice_indices = [
            min(max(b_idx + offset, 0), n_bscans - 1)
            for offset in range(-self.half_ctx, self.half_ctx + 1)
        ]
        slices = [scan['memmap'][s_idx] for s_idx in slice_indices]
        img_stack = np.stack(slices, axis=0).astype(np.float32)  # (C, 768, 320)

        # Normalize raw Solix 16-bit uint (0-2560) to [0.0, 1.0]
        img_stack = img_stack / 2560.0

        # 2. Extract Ground Truth Surfaces for central B-scan
        ilm_row = scan['curves']['ILM'][b_idx].copy()
        nfl_row = scan['curves']['NFL'][b_idx].copy()

        # 3. Generate Binary RNFL Mask: 1 inside RNFL, 0 outside (vectorized)
        rows, cols = img_stack.shape[1], img_stack.shape[2]
        cup_absent = np.isnan(nfl_row).astype(np.float32)

        y_coords = np.arange(rows, dtype=np.float32)[:, None]
        valid = ~np.isnan(ilm_row) & ~np.isnan(nfl_row) & (nfl_row > ilm_row)
        y0 = np.clip(np.round(np.nan_to_num(ilm_row, nan=-1)), 0, rows)
        y1 = np.clip(np.round(np.nan_to_num(nfl_row, nan=-1)), 0, rows)
        mask = ((y_coords >= y0) & (y_coords < y1) & valid).astype(np.float32)

        # 4. Standardize Eye Orientation: Flip OS horizontally so Nasal is consistently on one side
        if self.standardize_eye and scan['eye'] == 'OS':
            img_stack = np.flip(img_stack, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()
            ilm_row = np.flip(ilm_row).copy()
            nfl_row = np.flip(nfl_row).copy()
            cup_absent = np.flip(cup_absent).copy()

        # Fill NaNs in surface arrays with 0.0 for PyTorch tensor compatibility
        ilm_tensor = np.nan_to_num(ilm_row, nan=0.0).astype(np.float32)
        nfl_tensor = np.nan_to_num(nfl_row, nan=0.0).astype(np.float32)

        # Convert to PyTorch Tensors
        img_tensor = torch.from_numpy(img_stack)               # (C, 768, 320)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)      # (1, 768, 320)
        ilm_tensor = torch.from_numpy(ilm_tensor)              # (320,)
        nfl_tensor = torch.from_numpy(nfl_tensor)              # (320,)
        cup_absent_tensor = torch.from_numpy(cup_absent)       # (320,)

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'ilm_surface': ilm_tensor,
            'nfl_surface': nfl_tensor,
            'cup_absent': cup_absent_tensor,
            'subject': scan['subject'],
            'eye': scan['eye'],
            'bscan_idx': b_idx,
            'is_peripapillary': sample['is_peripapillary']
        }
