ckpt = "/root/Anti-UAV/sam_vit_b_01ec64.pth"
with open(ckpt, "rb") as f:
    print(f.read(16))  # 能读取前16字节吗？