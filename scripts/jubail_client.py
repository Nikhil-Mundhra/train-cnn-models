import pexpect
import sys

def run_jubail(commands, timeout=60):
    child = pexpect.spawn('ssh -o StrictHostKeyChecking=no nm4358@jubail.abudhabi.nyu.edu', timeout=timeout)
    child.expect('[Pp]assword:')
    child.sendline('Final.year@2026')
    child.expect(r'nm4358@login\d+:[^\n]*\$ ')
    
    results = []
    for cmd in commands:
        child.sendline(cmd)
        child.expect(r'nm4358@login\d+:[^\n]*\$ ')
        results.append(child.before.decode())
    
    child.sendline('exit')
    return results

if __name__ == '__main__':
    cmds = sys.argv[1:] if len(sys.argv) > 1 else ['squeue -u nm4358']
    res = run_jubail(cmds)
    for r in res:
        print(r)
