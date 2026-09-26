# Loss Formulation: Physics-Informed Multi-Task Objectives

The total training loss $\mathcal{L}_{\text{total}}$ is a composite multi-task objective ensuring dense volumetric accuracy, continuous sub-micron surface precision, anatomical cup validity, and physical optical edge alignment:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + w_{\text{bnd}} \mathcal{L}_{\text{bnd}} + w_{\text{cup}} \mathcal{L}_{\text{cup}} + w_{\text{topo}} \mathcal{L}_{\text{topo}} + w_{\text{edge}} \mathcal{L}_{\text{edge}} + w_{\text{thick}} \mathcal{L}_{\text{thick}}$$

---

### 1. Dense Segmentation Loss ($\mathcal{L}_{\text{seg}}$)

To prevent thin peripheral layers from vanishing under extreme class imbalance ($> 99.4\%$ background), $\mathcal{L}_{\text{seg}}$ combines **Focal Tversky Loss** with **Boundary-Weighted Cross-Entropy**:

$$\mathcal{L}_{\text{seg}} = w_{\text{dice}} \mathcal{L}_{\text{Tversky}} + w_{\text{bce}} \mathcal{L}_{\text{WeightedBCE}}$$

#### A. Asymmetric Focal Tversky Loss
$$\mathcal{L}_{\text{Tversky}} = 1 - \frac{\sum_i p_i y_i + \epsilon}{\sum_i p_i y_i + \alpha \sum_i p_i (1 - y_i) + \beta \sum_i (1 - p_i) y_i + \epsilon}$$
* We set **$\beta = 0.7$** (penalty for False Negatives / missed tissue) and **$\alpha = 0.3$** (penalty for False Positives).
* This forces the gradient to penalize missing a 4-pixel-tall peripheral layer far more than generating a slight boundary spill.

#### B. ReLayNet-Inspired Spatial Boundary Weighting
$$\mathcal{L}_{\text{WeightedBCE}} = \frac{1}{\Omega} \sum_{x,y} \omega(x,y) \cdot \text{BCE}\left(\sigma(\text{logit}_{x,y}),\; y_{x,y}\right)$$
where the spatial weight matrix $\omega(x,y)$ boosts transitions:
$$\omega(x,y) = 1.0 + 4.0 \cdot |\nabla y_{\text{mask}}| + 2.0 \cdot y_{\text{mask}}$$
* Pixels lying directly on the ILM or NFL layer boundaries receive **$5\times$ higher gradient updates**.
* Positive foreground voxels receive **$3\times$ higher gradient updates**.

---

### 2. Continuous 1D Boundary Regression Loss ($\mathcal{L}_{\text{bnd}}$)

Regresses the exact physical surface depth (in microns) along every A-scan column:
$$\mathcal{L}_{\text{ilm}} = \frac{1}{W} \sum_{x=1}^{W} \text{SmoothL1}\left(\hat{z}_{\text{ilm}}(x),\; z_{\text{ilm}}(x)\right) \cdot \Delta z_{\mu\text{m}}$$
$$\mathcal{L}_{\text{nfl}} = \frac{\sum_{x=1}^{W} \text{SmoothL1}\left(\hat{z}_{\text{nfl}}(x),\; z_{\text{nfl}}(x)\right) \cdot \Delta z_{\mu\text{m}} \cdot (1 - \text{cup\_absent}_x) \cdot w_{\text{peri}}}{\sum_{x=1}^{W} (1 - \text{cup\_absent}_x) \cdot w_{\text{peri}} + \epsilon}$$
$$\mathcal{L}_{\text{bnd}} = \mathcal{L}_{\text{ilm}} + \mathcal{L}_{\text{nfl}}$$
where $\Delta z_{\mu\text{m}} = 3.12367\,\mu\text{m/pixel}$.

---

### 3. Optic Cup Absence Loss ($\mathcal{L}_{\text{cup}}$)

Supervises the cup absence head over the BMO excavation:
$$\mathcal{L}_{\text{cup}} = \text{BCEWithLogits}\left(\hat{c}(x),\; c_{\text{absent}}(x)\right)$$

---

### 4. Topological Ordering Constraint ($\mathcal{L}_{\text{topo}}$)

Anatomically, the ILM (retinal surface) must always lie anterior to (above) the NFL boundary:
$$\mathcal{L}_{\text{topo}} = \frac{1}{W} \sum_{x=1}^{W} \max\left(0,\; \hat{z}_{\text{ilm}}(x) - \hat{z}_{\text{nfl}}(x)\right) \cdot (1 - \text{cup\_absent}_x)$$

---

### 5. Optical Reflectance Edge Alignment Loss ($\mathcal{L}_{\text{edge}}$)

Enforces that the continuous predicted NFL boundary snaps to the physical bright-to-dark drop-off in raw optical backscatter intensity:
$$\mathcal{G}(z, x) = \max\left(0,\; \mathbf{I}(z-1, x) - \mathbf{I}(z, x)\right) * \mathcal{K}_{\text{Gaussian}}$$
$$\mathcal{L}_{\text{edge}} = -\frac{\sum_{x=1}^{W} \text{SampleGrid}\left(\mathcal{G},\; \hat{z}_{\text{nfl}}(x), x\right) \cdot (1 - \text{cup\_absent}_x)}{\sum_{x=1}^{W} (1 - \text{cup\_absent}_x) + \epsilon}$$

---

### 6. Differentiable Column-Thickness Consistency Loss ($\mathcal{L}_{\text{thick}}$)

Ties the dense 2D mask directly to the 1D physical continuous thickness:
$$\hat{T}(x) = \sum_{y=1}^{H} \sigma\left(\text{logit}_{y, x}\right) \quad (\text{integrated column thickness in pixels})$$
$$T_{\text{gt}}(x) = \max\left(0,\; z_{\text{nfl}}(x) - z_{\text{ilm}}(x)\right) \cdot (1 - \text{cup\_absent}_x)$$
$$\mathcal{L}_{\text{thick}} = \frac{1}{\sum (1 - \text{cup\_absent}_x) + \epsilon} \sum_{x=1}^{W} \text{SmoothL1}\left(\hat{T}(x),\; T_{\text{gt}}(x)\right) \cdot \Delta z_{\mu\text{m}}$$

**Why this prevents mask cutoff**: If the dense logits drop toward zero in peripheral columns where the 1D surface heads indicate a $20\,\mu\text{m}$ layer, $\mathcal{L}_{\text{thick}}$ generates large positive gradients that force $\text{logit}_{y, x}$ back up until the integrated thickness matches $20\,\mu\text{m}$.
