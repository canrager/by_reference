from IPython.display import display, HTML
import matplotlib.pyplot as plt
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


## SVD

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
import einops
from tqdm import tqdm
from IPython.display import display, HTML


def plot_singular_vectors_interactive(
    act_LRBD, num_vs, layers, template_name, save_path=None, fixed_svd_layer=None
):
    """
    Plot singular vectors with optional fixed SVD from a specific layer.

    Args:
        act_LRBD: Activation tensor
        num_vs: Number of singular vectors to plot
        layers: List of layer indices
        template_name: Name of the task/template
        save_path: Path to save the HTML plot
        fixed_svd_layer: Layer index to use for fixed SVD basis (if None, compute SVD per layer)
    """
    num_relations = act_LRBD.shape[1]
    num_samples_per_relation = act_LRBD.shape[2]

    titles = [f"n={v}" for v in range(num_vs)]
    fig = make_subplots(rows=1, cols=num_vs, subplot_titles=titles)

    colors = px.colors.qualitative.Set1[:num_relations]

    # Compute fixed SVD if specified
    fixed_V = None
    if fixed_svd_layer is not None:
        if fixed_svd_layer not in layers:
            raise ValueError(f"fixed_svd_layer {fixed_svd_layer} must be in layers {layers}")
        act_BD = einops.rearrange(act_LRBD[fixed_svd_layer], "b r d -> (b r) d")
        _, _, fixed_V = torch.svd(act_BD)

    svd_results = []
    for layer_idx in tqdm(range(len(layers)), desc="Computing projections"):
        act_BD = einops.rearrange(act_LRBD[layer_idx], "b r d -> (b r) d")

        if fixed_V is None:
            _, _, V = torch.svd(act_BD)
        else:
            V = fixed_V

        svd_results.append(V)

    for layer_idx, layer in enumerate(layers):
        V = svd_results[layer_idx]

        for v in range(num_vs):
            for i in range(num_relations):
                pointer_components = act_LRBD[layer_idx][i] @ V.T[v]

                fig.add_trace(
                    go.Scatter(
                        x=torch.arange(num_samples_per_relation).numpy(),
                        y=pointer_components.numpy(),
                        mode="markers",
                        name=f"Pointer {i}",
                        legendgroup=f"Pointer {i}",
                        marker=dict(color=colors[i]),
                        showlegend=(v == 0),
                        visible=layer_idx == 0,
                    ),
                    row=1,
                    col=v + 1,
                )

            fig.update_xaxes(
                title_text="Samples" if v == num_vs // 2 else None, row=1, col=v + 1
            )
            fig.update_yaxes(
                title_text="V[n] @ resid_activations" if v == 0 else None, row=1, col=v + 1
            )

    steps = []
    for layer_idx in range(len(layers)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=f"Layer {layers[layer_idx]}"
            + (" (reference)" if layers[layer_idx] == fixed_svd_layer else ""),
        )
        for v in range(num_vs):
            for i in range(num_relations):
                trace_idx = layer_idx * (num_vs * num_relations) + v * num_relations + i
                step["args"][0]["visible"][trace_idx] = True
        steps.append(step)

    title_text = f"Projection of resid onto singular vector n, Task: {template_name}"
    if fixed_svd_layer is not None:
        title_text += f" (Fixed SVD from layer {fixed_svd_layer})"
    title_text += f"\nUse arrow keys to navigate layers; click on legend to toggle traces."

    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Layer: "},
            pad={"t": 50},
            steps=steps,
        )],
        title=title_text,
        height=500,
        showlegend=True,
    )

    # Improved arrow key navigation JavaScript
    arrow_key_js = """
    <script>
    (function() {
        function setupKeyboardNavigation() {
            // Find the Plotly div - more reliable selector
            var plotDivs = document.getElementsByClassName('plotly-graph-div');
            if (plotDivs.length === 0) {
                // If not found, try again after a short delay
                setTimeout(setupKeyboardNavigation, 100);
                return;
            }
            
            var plotDiv = plotDivs[0];
            
            function handleArrowKey(event) {
                if (!event.target.tagName.match(/input|textarea/i)) {  // Ignore if typing in input/textarea
                    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                        event.preventDefault();  // Prevent page scrolling
                        
                        var slider = plotDiv._fullLayout.sliders[0];
                        var currentStep = slider.active || 0;
                        var numSteps = slider.steps.length;
                        
                        // Calculate new step
                        var newStep = currentStep + (event.key === 'ArrowRight' ? 1 : -1);
                        newStep = Math.max(0, Math.min(newStep, numSteps - 1));
                        
                        if (newStep !== currentStep) {
                            // Get visibility settings from the step
                            var update = slider.steps[newStep].args[0].visible;
                            
                            // Update both the slider position and trace visibility
                            Plotly.update(plotDiv, 
                                {visible: update},  // Update trace visibility
                                {'sliders[0].active': newStep}  // Update slider position
                            );
                        }
                    }
                }
            }
            
            // Remove any existing listeners to prevent duplicates
            document.removeEventListener('keydown', handleArrowKey);
            // Add the event listener
            document.addEventListener('keydown', handleArrowKey);
        }
        
        // Initialize keyboard navigation
        if (document.readyState === 'complete') {
            setupKeyboardNavigation();
        } else {
            window.addEventListener('load', setupKeyboardNavigation);
        }
    })();
    </script>
    """

    if save_path is not None:
        # Convert figure to HTML and add the arrow key navigation
        fig_html = fig.to_html(full_html=True, include_plotlyjs=True)
        # Insert arrow key JavaScript before the closing </body> tag
        fig_html = fig_html.replace("</body>", f"{arrow_key_js}</body>")
        with open(save_path, "w") as f:
            f.write(fig_html)
    else:
        # For notebook display
        display(HTML(arrow_key_js))
        fig.show()

    return fig


def plot_subspace_patching_multiple_tasks(
    v_outcomes_ds, labels, title, save_path=None
):
    # Separate singular and random outcomes
    singular_outcomes = {k: v["singular"] for k, v in v_outcomes_ds.items()}
    random_outcomes = {k: v["random"] for k, v in v_outcomes_ds.items()}

    # Normalize other_base_object by 3
    for k in singular_outcomes.keys():
        singular_outcomes[k][labels.index("other_objects") - 1] /= 3
        random_outcomes[k][labels.index("other_objects") - 1] /= 3

    # Calculate percentages
    proportions = {k: v / v.sum() for k, v in singular_outcomes.items()}
    random_proportions = {k: v / v.sum() for k, v in random_outcomes.items()}

    # Set up the plot with adjusted figure size to accommodate external legend
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)

    # Calculate bar positions
    n_groups = len(proportions.keys())
    n_bars = len(labels)
    group_width = 0.8
    bar_width = group_width / n_bars

    # Create bars grouped by k
    random_line_added_to_legend = False
    for i, label in enumerate(labels):
        positions = np.arange(n_groups) + i * bar_width
        values = [proportions[k][i] for k in proportions.keys()]
        random_values = [random_proportions[k][i] for k in random_proportions.keys()]

        bars = ax.bar(positions, values, bar_width, label=label)
        # Add random baseline lines
        for pos, rand_val in zip(positions, random_values):
            if not random_line_added_to_legend:
                ax.plot(
                    [pos - bar_width / 2, pos + bar_width / 2],
                    [rand_val, rand_val],
                    color="black",
                    linewidth=2,
                    label="Random vector patch",
                )
                random_line_added_to_legend = True
            else:
                ax.plot(
                    [pos - bar_width / 2, pos + bar_width / 2],
                    [rand_val, rand_val],
                    color="black",
                    linewidth=2,
                )

    # Adjust x-axis
    plt.xticks(np.arange(n_groups) + (group_width / 2 - bar_width / 2), proportions.keys())

    plt.title(title)
    plt.ylabel(f"Proportion of {int(sum(singular_outcomes['box']))} interventions")
    plt.ylim(0, 1)
    plt.xlabel("Intervened template")

    # Add baseline accuracy line
    plt.axhline(y=0.2, color="r", linestyle="--", label="Random object prediction")

    # Move legend outside and to the upper right
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Intervention behavior")

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    # Adjust subplot parameters to give specified padding
    plt.subplots_adjust(right=0.85)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_mean_svd_scalar_per_attribute(
    relation_scalar_LRC,
    relation_scalar_errors_LRC,  # New parameter for errors
    singular_dims_C,
    layers,
    num_relations,
    save_path=None,
    fontsize=16,
):
    """
    Plot relation scalars across layers with error bands in a two-column grid layout.
    
    Args:
        relation_scalar_LRC: Tensor of shape (num_layers, num_relations, num_components)
        relation_scalar_errors_LRC: Tensor of shape (num_layers, num_relations, num_components)
                                  containing standard errors or confidence intervals
        singular_dims_C: List of component indices/dimensions
        layers: List of layer indices
        num_relations: Number of relations/pointers to plot
    """
    # Calculate number of rows needed for two columns
    num_plots = len(singular_dims_C)
    num_rows = (num_plots + 1) // 2  # Ceiling division to handle odd numbers
    
    # Create figure with two columns
    fig, axs = plt.subplots(num_rows, 2, figsize=(16, 5 * num_rows))
    fig.tight_layout(pad=10.0)
    
    # Convert axs to 2D array if there's only one row
    if num_rows == 1:
        axs = np.array([axs])
    
    # Flatten axs for easier iteration
    axs_flat = axs.flatten()
    
    # Define colors for different pointers
    colors = plt.cm.tab10(np.linspace(0, 1, num_relations))
    
    for j_idx, j in enumerate(singular_dims_C):
        ax = axs_flat[j_idx]
        
        # Plot each relation line with error band
        for i in range(num_relations):
            # Extract data and errors for current relation and component
            data = relation_scalar_LRC[:, i, j_idx].numpy()
            errors = relation_scalar_errors_LRC[:, i, j_idx].numpy()
            x = np.arange(len(data))
            
            # Plot error band
            ax.fill_between(
                x,
                data - errors,
                data + errors,
                alpha=0.2,
                color=colors[i],
                label=f"_nolegend_"  # Prevents error bands from appearing in legend
            )
            
            # Plot main line
            ax.plot(
                x, 
                data,
                label=f"Pointer {i}",
                color=colors[i],
                linewidth=2.5
            )
        
        # Increase font sizes
        ax.set_title(f"Projection onto V[{j}]", fontsize=fontsize, pad=15)
        ax.set_xlabel("Layer", fontsize=fontsize, labelpad=10)
        ax.set_ylabel("Projection scalar", fontsize=fontsize, labelpad=10)
        
        # Enhance grid and axis appearance
        ax.grid(True, linestyle='--', alpha=1)
        ax.axhline(0, color="black", linestyle="-", linewidth=1.5)
        ax.set_xlim(0, len(layers) - 1)
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
        # Enhanced legend
        ax.legend(fontsize=11)
        
        # Add some padding to the y-axis limits
        y_min, y_max = ax.get_ylim()
        padding = (y_max - y_min) * 0.1  # 10% padding
        ax.set_ylim(y_min - padding, y_max + padding)
    
    # Remove any empty subplots if odd number of components
    if num_plots % 2 == 1:
        axs_flat[-1].remove()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()


## Cosine similarities with singular vectors for each token

def prepare_activations(model, base_sample, v_DC):
    """Prepare model activations and compute cosine similarities."""
    # Get base text and tokenize
    base_text = base_sample["text"]
    encoding = model.tokenizer(base_text)
    tokens = model.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    
    # Get activations for each layer
    layers = range(model._model.config.num_hidden_layers)
    resid_post_LBPD = []
    
    with torch.no_grad(), model.trace(base_text):
        for layer_idx, layer in enumerate(layers):
            resid_post_module = model.model.layers[layer]
            resid_post_LBPD.append(resid_post_module.output[0].save())
    
    # Concatenate and normalize
    resid_post_LPD = torch.cat(resid_post_LBPD, dim=0)
    resid_post_LPD = resid_post_LPD / resid_post_LPD.norm(dim=-1, keepdim=True)
    resid_post_LPD = resid_post_LPD.to(torch.float).cpu()
    
    return resid_post_LPD, tokens

def plot_svd_cosine_similarities_per_token_over_layers_interactive(model, base_sample, v_DC, save_path=None):
    """Create interactive Plotly plot of cosine similarities across all layers."""
    # Prepare activations
    resid_post_LPD, tokens = prepare_activations(model, base_sample, v_DC)
    resid_post_LPD = resid_post_LPD.to(torch.float)
    v_DC = v_DC.to(torch.float)
    num_layers = len(resid_post_LPD)
    
    # Initialize figure
    fig = go.Figure()
    
    # Normalize direction vectors
    v1 = v_DC[:, 0] / torch.norm(v_DC[:, 0])
    v2 = v_DC[:, 1] / torch.norm(v_DC[:, 1])
    
    # Get positions for different token groups
    entity_positions = [base_sample['key_to_position'][f'e{i}'] for i in range(5)]
    attribute_positions = [base_sample['key_to_position'][f'a{i}'] for i in range(5)]
    question_position = [base_sample['key_to_position']['a_question']]
    the_positions = [i for i, token in enumerate(tokens) 
                    if token.lower().strip("▁ ") == 'the']
    box_positions = [i for i, token in enumerate(tokens) 
                    if token.lower().strip("▁ ") == 'box']
    
    # Create set of special positions
    special_positions = set(entity_positions + attribute_positions + 
                          the_positions + box_positions + question_position)
    
    # Get other positions (excluding padding tokens)
    other_positions = [i for i, token in enumerate(tokens) 
                      if i not in special_positions and token.strip() != '<pad>']
    
    # Create traces for each layer
    for layer_idx in range(num_layers):
        cos_sim_v1 = resid_post_LPD[layer_idx] @ v1
        cos_sim_v2 = resid_post_LPD[layer_idx] @ v2
        
        # Add traces for each token group
        traces = [
            go.Scatter(
                x=cos_sim_v1[entity_positions].numpy(),
                y=cos_sim_v2[entity_positions].numpy(),
                mode='markers+text',
                name='Entities',
                text=[tokens[pos] for pos in entity_positions],
                textposition="top center",
                marker=dict(size=12, color='blue'),
                visible=(layer_idx == 0),
                showlegend=True,  # Always show legend
                legendgroup='entities',
                legendgrouptitle_text='Token types'
            ),
            go.Scatter(
                x=cos_sim_v1[attribute_positions].numpy(),
                y=cos_sim_v2[attribute_positions].numpy(),
                mode='markers+text',
                name='Attributes',
                text=[tokens[pos] for pos in attribute_positions],
                textposition="top center",
                marker=dict(size=12, color='red'),
                visible=(layer_idx == 0),
                showlegend=True,
                legendgroup='attributes'
            ),
            go.Scatter(
                x=cos_sim_v1[question_position].numpy(),
                y=cos_sim_v2[question_position].numpy(),
                mode='markers+text',
                name='Question Box Label',
                text=[tokens[pos] for pos in question_position],
                textposition="top center",
                marker=dict(size=15, color='orange', symbol='star'),
                visible=(layer_idx == 0),
                showlegend=True,
                legendgroup='question'
            ),
            go.Scatter(
                x=cos_sim_v1[the_positions].numpy(),
                y=cos_sim_v2[the_positions].numpy(),
                mode='markers',
                name='"the" tokens',
                marker=dict(size=10, color='green'),
                visible=(layer_idx == 0),
                showlegend=True,
                legendgroup='the'
            ),
            go.Scatter(
                x=cos_sim_v1[box_positions].numpy(),
                y=cos_sim_v2[box_positions].numpy(),
                mode='markers',
                name='"box" tokens',
                marker=dict(size=10, color='purple'),
                visible=(layer_idx == 0),
                showlegend=True,
                legendgroup='box'
            ),
            go.Scatter(
                x=cos_sim_v1[other_positions].numpy(),
                y=cos_sim_v2[other_positions].numpy(),
                mode='markers',
                name='Other tokens',
                marker=dict(size=8, color='gray', opacity=0.5),
                visible=(layer_idx == 0),
                showlegend=True,
                legendgroup='other'
            )
        ]
        
        for trace in traces:
            fig.add_trace(trace)
    
    # Create steps for slider
    steps = []
    num_traces_per_layer = len(traces)
    for layer_idx in range(num_layers):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=f"Layer {layer_idx}"
        )
        # Make traces for current layer visible
        for trace_idx in range(num_traces_per_layer):
            step["args"][0]["visible"][layer_idx * num_traces_per_layer + trace_idx] = True
        steps.append(step)
    
    # Calculate global min/max for axis ranges
    x_coords = []
    y_coords = []
    for layer_idx in range(num_layers):
        cos_sim_v1 = resid_post_LPD[layer_idx] @ v1
        cos_sim_v2 = resid_post_LPD[layer_idx] @ v2
        x_coords.extend(cos_sim_v1.numpy())
        y_coords.extend(cos_sim_v2.numpy())
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add some padding to the ranges (5% on each side)
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    
    # Update layout
    fig.update_layout(
        title="Cosine Similarities of activations with singular vectors<br>Use arrow keys or slider to navigate layers<br>Scroll down to see the full sentence",
        xaxis_title="Cosine Similarity with V[1]",
        yaxis_title="Cosine Similarity with V[2]",
        xaxis=dict(
            range=[x_min - x_padding, x_max + x_padding],
            scaleanchor="y",
            scaleratio=1,
            fixedrange=True  # Prevent zoom/pan
        ),
        yaxis=dict(
            range=[y_min - y_padding, y_max + y_padding],
            fixedrange=True  # Prevent zoom/pan
        ),
        uirevision=True,  # Maintain UI state across updates
        width=800,
        height=800,
        showlegend=True,
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Layer: "},
            pad={"t": 50},
            steps=steps
        )],
    )

    # Create tokens display div
    filtered_tokens = [token for token in tokens if token.strip() != '<pad>']
    token_spans = []
    for i, token in enumerate(filtered_tokens):
        # Determine token type and color
        color = 'gray'  # default color
        if any(i == pos for pos in entity_positions):
            color = 'blue'
        elif any(i == pos for pos in attribute_positions):
            color = 'red'
        elif any(i == pos for pos in question_position):
            color = 'orange'
        elif any(i == pos for pos in the_positions):
            color = 'green'
        elif any(i == pos for pos in box_positions):
            color = 'purple'
            
        token_spans.append(f'<span style="color: {color}; margin: 0 2px;">{token}</span>')
    
    tokens_div = f"""
    <div id="tokens-display" style="
        margin-top: 20px;
        padding: 15px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 14px;
        line-height: 1.6;
        overflow-x: auto;
        white-space: nowrap;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    ">
        <div style="font-weight: bold; margin-bottom: 8px; color: #495057;">Token Sequence:</div>
        <div style="padding: 8px; background-color: white; border-radius: 4px;">
            {''.join(token_spans)}
        </div>
    </div>
    """

    arrow_key_js = """
    <script>
    (function() {
        // Function to set up keyboard navigation
        function setupKeyboardNavigation() {
            var plotDivs = document.getElementsByClassName('plotly-graph-div');
            if (plotDivs.length === 0) {
                setTimeout(setupKeyboardNavigation, 100);
                return;
            }
            
            var plotDiv = plotDivs[0];
            
            // Handle arrow key events
            function handleArrowKey(event) {
                // Only handle arrow keys if not in an input/textarea
                if (!event.target.tagName.match(/input|textarea/i)) {
                    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                        event.preventDefault();
                        
                        // Get current slider state
                        var slider = plotDiv._fullLayout.sliders[0];
                        var currentStep = slider.active || 0;
                        var numSteps = slider.steps.length;
                        
                        // Calculate new step
                        var newStep = currentStep + (event.key === 'ArrowRight' ? 1 : -1);
                        newStep = Math.max(0, Math.min(newStep, numSteps - 1));
                        
                        // Update plot if step changed
                        if (newStep !== currentStep) {
                            var update = slider.steps[newStep].args[0].visible;
                            
                            Plotly.update(plotDiv, 
                                {visible: update},
                                {'sliders[0].active': newStep}
                            );
                        }
                    }
                }
            }
            
            // Set up event listeners
            document.removeEventListener('keydown', handleArrowKey);
            document.addEventListener('keydown', handleArrowKey);
        }
        
        // Initialize when document is ready
        if (document.readyState === 'complete') {
            setupKeyboardNavigation();
        } else {
            window.addEventListener('load', setupKeyboardNavigation);
        }
    })();
    </script>
    """

    if save_path is not None:
        # Generate and save HTML
        fig_html = fig.to_html(full_html=True, include_plotlyjs=True)
        fig_html = fig_html.replace("</body>", f"{tokens_div}{arrow_key_js}</body>")
        with open(save_path, "w") as f:
            f.write(fig_html)
    else:
        # Display in notebook
        display(HTML(arrow_key_js))
        fig.show()
    
    return fig