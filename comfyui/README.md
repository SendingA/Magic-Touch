### Forwarding
```bash
ssh -v -N -L 0.0.0.0:8188:l40gpu001:8188 cse12110817@172.18.34.25 -p10022
# or
ssh -v -N -L 0.0.0.0:8188:a100gpu003:8188 cse12110817@172.18.34.25 -p10022
```