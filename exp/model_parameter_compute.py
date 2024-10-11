def compute_model(model, require_grad):
    if require_grad:
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 计算可训练参数数量
    else:
        model_size = sum(p.numel() for p in model.parameters())
    model_size_MB = (model_size * 4) / (1024 ** 2)  # 4字节/参数，转换为MB
    return f'Model size: {model_size_MB:.5f} MB'