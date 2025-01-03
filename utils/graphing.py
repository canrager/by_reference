from IPython.display import display, HTML
import numpy as np
import torch

def get_color_gradient(value, vmin, vmax):
    """
    Convert value to RGB color string:
    negative -> red
    zero -> white
    positive -> blue
    """
    abs_max = max(abs(vmin), abs(vmax))
    normalized = value / abs_max if abs_max != 0 else 0

    if normalized < 0:  # Red for negative
        r = 255
        g = b = int(255 * (1 + normalized))
    else:  # Blue for positive
        b = 255
        r = g = int(255 * (1 - normalized))

    return f"rgb({r}, {g}, {b})"


def get_text_color(value, abs_max):
    """Return black for abs(value) < 0.5, white for abs(value) >= 0.5"""
    return "white" if abs(value) >= 0.5 * abs_max else "black"


def visualize_token_activations(tokens, activations):
    """
    Visualize tokens as colored inline rectangles with their activation values.
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()

    if len(tokens) != len(activations):
        raise ValueError(
            f"Length mismatch: {len(tokens)} tokens vs {len(activations)} activation values"
        )

    vmin, vmax = np.min(activations), np.max(activations)
    abs_max = max(abs(vmin), abs(vmax))

    scale_ref = f"""
        <div style='margin-bottom: 20px;'>
            <span style='margin-right: 15px; font-size: 0.9em;'>
                Scale: 
                <span style='background-color: {get_color_gradient(vmin, vmin, vmax)}; 
                           color: {get_text_color(vmin, abs_max)}; 
                           padding: 2px 8px; 
                           border-radius: 3px;'>
                    min: {vmin:.3f}
                </span>
                <span style='background-color: white; 
                           color: black; 
                           padding: 2px 8px; 
                           border-radius: 3px; 
                           margin: 0 5px;'>
                    0.000
                </span>
                <span style='background-color: {get_color_gradient(vmax, vmin, vmax)}; 
                           color: {get_text_color(vmax, abs_max)}; 
                           padding: 2px 8px; 
                           border-radius: 3px;'>
                    max: {vmax:.3f}
                </span>
            </span>
        </div>
    """

    tokens_html = ""
    for token, value in zip(tokens, activations):
        bg_color = get_color_gradient(value, vmin, vmax)
        text_color = get_text_color(value, abs_max)

        tokens_html += f"""
            <span class='token-box' style='background-color: {bg_color}; color: {text_color};'>
                <span class='activation-value' style='color: #666;'>{value:.3f}</span>
                {token}
            </span>
        """

    html_output = f"""
    <div style='font-family: monospace; line-height: 2.5; background-color: white; padding: 20px;'>
        <style>
            .token-box {{
                display: inline-block;
                padding: 4px 8px;
                margin: 2px;
                border-radius: 3px;
                font-weight: bold;
                position: relative;
            }}
            .activation-value {{
                position: absolute;
                top: -18px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 0.8em;
                white-space: nowrap;
            }}
        </style>
        {scale_ref}
        {tokens_html}
    </div>
    """

    display(HTML(html_output))