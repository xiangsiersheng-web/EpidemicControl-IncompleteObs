def mask_params_to_str(args) -> str:
    args_dict = vars(args)
    if "mask_rate_down" not in args_dict:
        return ""
    mask_str = (f"_{float(args.mask_rate_down):.2f}_{float(args.mask_rate_up):.2f}"
                f"_{int(args.mask_duration_down)}_{args.mask_duration_up}")
    # 去除字符中的小数点
    mask_str = mask_str.replace(".", "")
    return mask_str