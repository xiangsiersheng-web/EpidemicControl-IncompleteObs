"""Small shared helpers required by the public evaluation entry points."""

def mask_params_to_str(args) -> str:
    values = vars(args)
    if "mask_rate_down" not in values:
        return ""
    text = (f"_{float(args.mask_rate_down):.2f}_{float(args.mask_rate_up):.2f}"
            f"_{int(args.mask_duration_down)}_{args.mask_duration_up}")
    return text.replace(".", "")
