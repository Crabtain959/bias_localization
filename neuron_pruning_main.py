#!/usr/bin/env python
# coding: utf-8

# # Neuron Scoring

# This script contains all functions used for scoring and pruning neurons from a LLM using WandA when prompting the model with black and white prompts. This includes all model versions (with/without responses and specific/common pruning).

# In[ ]:


import transformers
import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict


# In[ ]:


def tokenize_prompts(prompts, tokenizer, model, responses=None):
    """
    Tokenizes prompts with or without responses using a tokenizer and model.

    Args:
        prompts (list): A list of prompts.
        tokenizer: The tokenizer to apply chat templates.
        model: The model to set the device for tensor processing.
        responses (list, optional): A list of responses corresponding to the prompts. Defaults to None.

    Returns:
        list: A list of tokenized prompts (and responses if provided).
    """
    tokenized_prompts = []

    for i, prompt in enumerate(prompts):
        if responses:
            # If responses are provided, include them in the messages
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": responses[i]}
            ]
        else:
            # If responses are not provided, only use the prompt
            messages = [
                {"role": "user", "content": prompt},
            ]

        input_tensor = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)
        
        tokenized_prompts.append(input_tensor)
    
    return tokenized_prompts


# In[ ]:


def prepare_calibration_input(model, tokenized_prompts, device):
    """
    Prepares input tensors for calibration by intercepting the inputs to the first layer of the model.

    Args:
        model: The pre-trained model whose inputs need to be prepared for calibration.
        tokenized_prompts (list): A list of tokenized prompts, each containing input tensors like 'input_ids' and 'attention_mask'.
        device (torch.device): The device (CPU or GPU) to which the model and inputs are to be moved.

    Returns:
        tuple: A tuple containing:
            - inps (list): A list of input tensors captured from the first layer.
            - outs (list): A list of None values, initialized for potential outputs.
            - attention_mask (list): A list of attention masks captured from the first layer.
            - position_ids (list): A list of position IDs captured from the first layer.
    """
    # Disable cache for model configuration to avoid using cached states
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Retrieve the model layers
    layers = model.model.layers

    # Adjust the device if the 'model.embed_tokens' layer is mapped to a different device
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]
        
    # Lists to store inputs, attention masks, and position IDs intercepted from the first layer
    inps = []
    attention_mask = []
    position_ids = []

    # A custom module to intercept inputs at the first layer
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_mask.append(kwargs["attention_mask"])
            position_ids.append(kwargs["position_ids"])
            # Raise an error to stop forward pass after capturing inputs
            raise ValueError

    # Replace the first layer of the model with the Catcher module to capture inputs
    layers[0] = Catcher(layers[0])
        
    # Process each batch of tokenized prompts
    for batch in tokenized_prompts:
        try:
            # Forward pass to capture inputs; no need for outputs
            model(input_ids=batch['input_ids'].to(device), 
                  attention_mask=batch['attention_mask'].to(device), 
                  position_ids=None)
            # Clear CUDA cache to manage memory usage
            torch.cuda.empty_cache()
        except ValueError:
            # Catch the ValueError to continue processing after input capture
            pass
        
    # Restore the original first layer of the model
    layers[0] = layers[0].module

    # Prepare an empty list for potential outputs (currently not used)
    outs = [None for _ in range(len(tokenized_prompts))]
    
    # Restore the original cache setting for the model
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


# In[ ]:


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively finds layers of a specified type(s) within a PyTorch module.

    Args:
        module (nn.Module): The PyTorch module to search within.
        layers (list): A list of layer types (e.g., [nn.Linear, nn.Conv2d]) to find.
        name (str): The base name for the module, used to construct full layer names.

    Returns:
        dict: A dictionary where keys are the names of the layers and values are the layer objects.
    """
    # Base case: If the current module's type is in the list of layers to find, return it
    if type(module) in layers:
        return {name: module}
    
    # Recursive case: Initialize an empty dictionary to store found layers
    res = {}
    
    # Iterate over all child modules
    for child_name, child_module in module.named_children():
        # Construct the full name for the child module
        full_name = name + "." + child_name if name != "" else child_name
        
        # Recursively search for layers in the child module and update the results dictionary
        res.update(find_layers(child_module, layers=layers, name=full_name))
    
    return res


# In[ ]:


class WrappedGPT:
    """
    This class wraps a GPT layer to perform specific operations, such as tracking
    activations, scaling inputs, and managing device placement.

    Attributes:
        layer (nn.Module): The GPT layer to be wrapped.
        dev (torch.device): The device where the layer's weight is stored.
        rows (int): The number of rows in the layer's weight matrix.
        columns (int): The number of columns in the layer's weight matrix.
        scaler_row (torch.Tensor): A tensor used to scale the input activations.
        activations (list): A list to store activations (currently unused).
        nsamples (int): The number of samples processed.
        layer_id (int): Identifier for the layer.
        layer_name (str): Name of the layer.
    """
    
    def __init__(self, layer, layer_id=0, layer_name="none"):
        """
        Initializes the WrappedGPT object with the provided layer and metadata.

        Args:
            layer (nn.Module): The GPT layer to be wrapped.
            layer_id (int): An identifier for the layer (default is 0).
            layer_name (str): A name for the layer (default is "none").
        """
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        # Initialize the scaler tensor and other attributes
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.activations = []  # Currently unused
        self.nsamples = 0

        # Metadata for the layer
        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        """
        Processes a batch of input and output data, updating the scaler and managing activations.

        Args:
            inp (torch.Tensor): Input tensor of shape (batch_size, seq_len) or (batch_size, seq_len, hidden_dim).
            out (torch.Tensor): Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        # Ensure input and output tensors have three dimensions
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        batch_size = inp.shape[0]

        # Check if the layer is a Linear layer and reshape input if necessary
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        # Update the scaler for the rows based on the number of samples processed
        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        # Convert input to float32 for numerical stability
        inp = inp.type(torch.float32)

        # Update scaler_row with the squared L2 norm of the input
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

        # Optionally store activations (currently disabled to save memory)
        # self.activations.append(inp)

        # Clean up to manage memory
        del inp
        torch.cuda.empty_cache()


# In[ ]:


def process_model_layers(df, model, tokenizer):
    """
    Processes model layers by tokenizing prompts, preparing calibration inputs,
    and calculating metrics for each layer based on input activations.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'variation', 'race', 'prompt_text', and 'response'.
        model (nn.Module): The pre-trained model to process.
        tokenizer: The tokenizer to use for encoding prompts and responses.
    """
    # Iterate through each group of data by variation and race
    for i, group in df.groupby(["variation", "race"]):
        # Disable caching for model configuration to avoid using cached states
        use_cache = model.config.use_cache
        model.config.use_cache = False
        
        # Extract variation and race
        variation = group.variation.iloc[0]
        race = group.race.iloc[0]
        
        # Tokenize prompts and responses
        inps_enc = tokenize_prompts(
            group.prompt_text.tolist(), 
            tokenizer, 
            model,
            group.response.tolist()
        )
        
        print(f"Starting for {variation} variation and race {race} with responses")
        print("Tokenization complete. Prompts are ready for further processing.")
        print(f"We have {len(inps_enc)} prompts.")
        
        # Prepare calibration inputs by passing tokenized prompts through the embedding layer
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, inps_enc, model.device)
        
        print("Calibration input prepared. Ready for scoring.")

        # Move input tensors to the correct device
        inps = [inp.squeeze(0).to(model.device) for inp in inps]
        attention_mask = [am.to(model.device) for am in attention_mask]
        position_ids = [pids.to(model.device) for pids in position_ids]
        layers = model.model.layers

        print("Inputs, attention masks, and position IDs are prepared and moved to the correct device.")

        # Process each layer in the model
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)  # Find all relevant sublayers
            
            if f"model.layers.{i}" in model.hf_device_map:
                # Handle multi-GPU cases
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps = [inp.to(dev) for inp in inps]
                outs = [out.to(dev) if out is not None else out for out in outs]
                attention_mask = [am.to(dev) for am in attention_mask]
                position_ids = [pids.to(dev) for pids in position_ids]

            # Wrap layers with WrappedGPT for activation tracking
            wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

            def add_batch(name):
                """Helper function to add a batch for each wrapped layer."""
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            # Process inputs through the current layer and register forward hooks
            for j in range(len(inps)):
                handles = []
                for name in wrapped_layers:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))

                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask[j],
                        position_ids=position_ids[j],
                    )[0]

                for h in handles:
                    h.remove()

            # Calculate and save scores for each sublayer
            for name in subset:
                print(f"Scoring layer {i} name {name}")

                magnitude = torch.abs(subset[name].weight.data)
                act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_metric = magnitude * act

                # Save the calculated scores
                save_folder = os.path.join(f"scores_all/wanda_scores_{variation}_with_responses/{race}_weights_{variation}_with_responses")
                os.makedirs(save_folder, exist_ok=True)

                target_file = os.path.join(
                    save_folder,
                    f"W_metric_layer_{i}_name_{name}_{race}_weights_{variation}_with_responses.pkl",
                )

                with open(target_file, "wb") as f:
                    print(
                        f"Writing W_metric in layer {i} and name {name} with {race} prompts, {variation} variation with responses to the file"
                    )
                    pickle.dump(W_metric, f)
        
        # Swap inps and outs for further processing
        for j in range(len(inps)):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0].squeeze(0)
        inps, outs = outs, inps

        # Restore original cache setting and clear CUDA cache
        model.config.use_cache = use_cache
        torch.cuda.empty_cache()


# In[ ]:


map_layers = {
    'q' : 'self_attn.q_proj',
    'k' : 'self_attn.k_proj',
    'v' : 'self_attn.v_proj',
    'o' : 'self_attn.o_proj',
    'gate' : 'mlp.gate_proj',
    'up' : 'mlp.up_proj',
    'down' : 'mlp.down_proj'
}


# In[ ]:


def prune_from_training_common_neurons(model, tokenizer, csv_file):
    """
    Prunes neurons from a model's layers based on training data of common neurons across a training set of prompts provided in a CSV file.

    Args:
        model (nn.Module): The pre-trained model whose neurons are to be pruned.
        tokenizer: The tokenizer associated with the model (not used directly in this function but included for consistency).
        csv_file (str): Path to the CSV file containing columns: 'layer_num', 'sublayer', 'neuron_index', and 'num_prompts'.

    Returns:
        model (nn.Module): The pruned model.
        tokenizer: The unchanged tokenizer.
    """
    # Save the model's original cache configuration and disable caching to prevent using cached states
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Read and preprocess the CSV file
    df = pd.read_csv(csv_file)
    df = df[df["num_prompts"] == 12].copy()  # Filter rows where 'num_prompts' equals 12, that is those neurons which appear in every training prompt

    # Ensure 'layer_num' and 'neuron_index' are treated as integers
    df['layer_num'] = df['layer_num'].astype(int)
    df["neuron_index"] = df["neuron_index"].apply(lambda x: int(x.split('.')[2]))

    # Group the DataFrame by 'layer_num' and 'sublayer' to process each combination
    grouped = df.groupby(['layer_num', 'sublayer'])

    # Process each group of layers and sublayers
    for (layer_num, name_layer_alias), group in grouped:
        # Retrieve the specific layer and its subset for pruning
        layer = layers[layer_num]
        subset = find_layers(layer)
        
        # Map the alias to the actual name of the sublayer
        name_layer = map_layers[name_layer_alias]
        
        # Skip if the sublayer is not in the subset
        if name_layer not in subset:
            continue
        
        print(f"Processing layer {layer_num}, sublayer {name_layer}")

        # Get the neuron indices to prune for this specific layer and sublayer
        prune_indices = group['neuron_index'].values
        
        # Calculate the rows and columns for the indices to prune
        weight_dim = subset[name_layer].weight.data.shape[1]
        prune_rows = prune_indices // weight_dim
        prune_cols = prune_indices % weight_dim

        # Create a mask to set the weights of the pruned neurons to zero
        W_mask = torch.zeros_like(subset[name_layer].weight.data) == 1
        W_mask[prune_rows, prune_cols] = True
        subset[name_layer].weight.data[W_mask] = 0  # Prune weights by setting them to zero

        # Calculate and print the percentage of pruned weights
        total_weights = subset[name_layer].weight.data.numel()
        num_pruned_weights = len(prune_indices)
        prune_ratio = num_pruned_weights / total_weights
        print(f"Layer {layer_num} name {name_layer}: Pruned {num_pruned_weights} weights ({prune_ratio:.4%} of total weights)")

    # Restore the model's original cache configuration
    model.config.use_cache = use_cache

    return model, tokenizer


# In[ ]:


def prune_setDiff(model, tokenizer, item, model_version, top_white_percent=0.15, top_black_percent=0.15):
    """
    Prunes neurons from a model's layers based on the set difference between the top neurons
    identified from two different groups (e.g., 'white' and 'black') for a given item (variation).

    Args:
        model (nn.Module): The pre-trained model whose neurons are to be pruned.
        tokenizer: The tokenizer associated with the model (not used directly in this function but included for consistency).
        item (str): The specific item/variation (e.g., 'chair') for which the pruning is performed.
        model_version (str): Either 'black' or 'white' to define from which groups are the top neurons.
        top_white_percent (float): The percentage of top neurons to select from the 'white' group.
        top_black_percent (float): The percentage of top neurons to select from the 'black' group.

    Returns:
        model (nn.Module): The pruned model.
        tokenizer: The unchanged tokenizer.
    """
    # Save the model's original cache configuration and disable caching to prevent using cached states
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    print(f"Pruning based on top {top_white_percent * 100}% white and top {top_black_percent * 100}% black neurons for {item} with responses.")

    # Iterate through each layer of the model
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)  # Get all relevant sublayers
        
        # Process each sublayer within the current layer
        for name in subset:
            print(f"Processing layer {i} name {name}")
            
            # Load the W_metric scores for both 'white' and 'black' categories
            W_metric_white = pickle.load(
                open(f"scores_all/wanda_scores_{item}_with_responses/white_weights_{item}_with_responses/W_metric_layer_{i}_name_{name}_white_weights_{item}_with_responses.pkl", "rb")
            )
            W_metric_black = pickle.load(
                open(f"scores_all/wanda_scores_{item}_with_responses/black_weights_{item}_with_responses/W_metric_layer_{i}_name_{name}_black_weights_{item}_with_responses.pkl", "rb")
            )

            # Ensure the tensors are moved to CPU before converting to NumPy for processing
            W_metric_white_cpu = W_metric_white.cpu().numpy()
            W_metric_black_cpu = W_metric_black.cpu().numpy()

            # Flatten the arrays to work with them easily
            W_metric_white_flat = W_metric_white_cpu.flatten()
            W_metric_black_flat = W_metric_black_cpu.flatten()

            # Select top % of 'white' neurons
            num_top_white = int(top_white_percent * W_metric_white_flat.size)
            top_white_indices = torch.topk(torch.tensor(W_metric_white_flat), num_top_white, largest=True)[1].numpy()

            # Select top % of 'black' neurons
            num_top_black = int(top_black_percent * W_metric_black_flat.size)
            top_black_indices = torch.topk(torch.tensor(W_metric_black_flat), num_top_black, largest=True)[1].numpy()

            # Find the set difference between the top 'white' and 'black' neurons
            if model_version == "black":
                prune_indices = np.setdiff1d(top_black_indices, top_white_indices)
            else:
                prune_indices = np.setdiff1d(top_white_indices, top_black_indices)

            # Determine the rows and columns of the indices to prune
            weight_dim = subset[name].weight.data.shape[1]
            prune_rows = prune_indices // weight_dim
            prune_cols = prune_indices % weight_dim

            # Create a mask to set the weights of the pruned neurons to zero
            W_mask = torch.zeros_like(subset[name].weight.data) == 1
            W_mask[prune_rows, prune_cols] = True
            subset[name].weight.data[W_mask] = 0  # Prune weights by setting them to zero

            # Calculate and print the percentage of pruned weights
            total_weights = subset[name].weight.data.numel()
            num_pruned_weights = len(prune_indices)
            prune_ratio = num_pruned_weights / total_weights
            print(f"Layer {i} name {name}: Pruned {num_pruned_weights} weights ({prune_ratio:.4%} of total weights) with responses")

    # Restore the model's original cache configuration
    model.config.use_cache = use_cache

    return model, tokenizer


# In[ ]:


def get_inputs(prompt, terminators, tokenizer, model):
    """
    Generates tokenized inputs for a given prompt using a specified tokenizer and model.

    Args:
        prompt (str): The prompt text to be tokenized.
        terminators (list): List of terminators for the prompt (not used in this function, but included for potential future use).
        tokenizer: The tokenizer to apply the chat template for tokenization.
        model: The model to determine the device where the inputs should be moved.

    Returns:
        inputs (torch.Tensor): Tokenized inputs ready for model processing.
    """
    # Create a list of messages for the tokenizer's chat template
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Apply the tokenizer's chat template to convert messages to tokenized input tensors
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)  # Move the input tensors to the model's device

    return inputs


# In[ ]:


def generate_responses(df, pruned_model, pruned_tokenizer, terminators, num_iterations=100, temperature=0.6):
    """
    Generates responses for a DataFrame of prompts using a pruned model and tokenizer.

    Args:
        df (pd.DataFrame): DataFrame containing prompt data with necessary columns.
        pruned_model (nn.Module): The pruned model used for generating responses.
        pruned_tokenizer: The tokenizer associated with the pruned model.
        terminators (list): List of terminator tokens for the generation process.
        num_iterations (int): Number of iterations to repeat the generation process. Default is 100.
        temperature (float): The temperature parameter for sampling during generation. Default is 0.6.

    Returns:
        list: A list of dictionaries, each containing the scenario, variation, name group, name, context level, prompt text, and generated response.
    """
    outputs = []

    # Repeat the process for the specified number of iterations
    for _ in tqdm(range(num_iterations), desc="Iterations"):
        # Iterate over each row in the DataFrame
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Responses"):
            prompt = row['prompt_text']  # Extract the prompt text from the current row

            # Generate tokenized inputs using the provided tokenizer
            input_ids = get_inputs(prompt, terminators, pruned_tokenizer, pruned_model)

            # Generate output using the pruned model
            output = pruned_model.generate(
                input_ids,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
            )

            # Decode the generated output after the prompt
            response = output[0][input_ids.shape[-1]:]
            decoded = pruned_tokenizer.decode(response, skip_special_tokens=True)

            # Prepare the output dictionary for the current prompt and response
            output_dict = {
                "scenario": row["scenario"],
                "variation": row["variation"],
                "name_group": row["name_group"],
                "name": row["name"],
                "context_level": row["context_level"],
                "prompt_text": row["prompt_text"],
                "response": decoded
            }

            # Append the output dictionary to the outputs list
            outputs.append(output_dict)

    return outputs


# In[ ]:


def create_df_top_neurons_setDiff(model, variations, model_version, folder="scores_all", top_p_percent=0.15):
    """
    Creates a DataFrame-like structure containing the top neurons' indices for each layer's sublayer,
    based on the set difference between the top neurons identified for two groups ('white' and 'black').

    Args:
        model (nn.Module): The pre-trained model to process.
        variations (list): List of variations/items to process.
        model_version (str): Either 'black' or 'white' to define from which groups are the top neurons.
        folder (str): Directory containing the scores files. Default is "scores_all".
        top_p_percent (float): The percentage of top neurons to select. Default is 0.15 (15%).

    """
    # Disable caching for model configuration to avoid using cached states
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Iterate through each variation in the provided list
    for variation in tqdm(variations, desc="Processing items"):
        rows = []  # List to store data for the DataFrame

        # Iterate through each layer in the model
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)  # Find all relevant sublayers

            # Process each sublayer within the current layer
            for name in subset:
                print(f"Processing variation {variation}, both races, layer {i}, sublayer {name} with responses")

                # Paths to the 'white' and 'black' weights metrics
                path_white = f"W_metric_layer_{i}_name_{name}_white_weights_{variation}_with_responses.pkl"
                path_black = f"W_metric_layer_{i}_name_{name}_black_weights_{variation}_with_responses.pkl"

                # Load the metric scores for both 'white' and 'black' categories
                W_metric_white = pickle.load(
                    open(f"{folder}/wanda_scores_{variation}_with_responses/white_weights_{variation}_with_responses/{path_white}", "rb")
                )
                W_metric_black = pickle.load(
                    open(f"{folder}/wanda_scores_{variation}_with_responses/black_weights_{variation}_with_responses/{path_black}", "rb")
                )

                # Ensure tensors are moved to CPU before converting to NumPy
                W_metric_white_cpu = W_metric_white.cpu().numpy()
                W_metric_black_cpu = W_metric_black.cpu().numpy()

                # Flatten the arrays to work with them easily
                W_metric_white_flat = W_metric_white_cpu.flatten()
                W_metric_black_flat = W_metric_black_cpu.flatten()

                # Select top % of 'white' neurons
                num_top_white = int(top_p_percent * W_metric_white_flat.size)
                top_white_indices = torch.topk(torch.tensor(W_metric_white_flat), num_top_white, largest=True)[1].numpy()

                # Select top % of 'black' neurons
                num_top_black = int(top_p_percent * W_metric_black_flat.size)
                top_black_indices = torch.topk(torch.tensor(W_metric_black_flat), num_top_black, largest=True)[1].numpy()

                # Find the set difference between the top 'black' and 'white' neurons
                if model_version == "black":
                    prune_indices = np.setdiff1d(top_black_indices, top_white_indices)
                else:
                    prune_indices = np.setdiff1d(top_white_indices, top_black_indices)

                # Add the top neurons to the list
                for idx in prune_indices:
                    df_row = {
                        "layer": i,
                        "sublayer": name,
                        "neuron_index": idx
                    }
                    rows.append(df_row)

        # Save the results to a pickle file after processing each variation
        save_path = f"{variation}_pruned_{model_version}_{int(top_p_percent * 100)}_with_responses.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(rows, f)
    
    # Restore the model's original cache configuration
    model.config.use_cache = use_cache


# In[ ]:


def open_pickle(file):
    """
    Opens a pickle file and loads its content.

    Args:
        file (str): Path to the pickle file.

    Returns:
        data (list): Data loaded from the pickle file.
    """
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


# In[ ]:


def transform_data(data):
    """
    Transforms the data to extract neuron information in a specific format.

    Args:
        data (list): List of dictionaries containing neuron information.

    Returns:
        transformed_data (list): List of dictionaries with formatted neuron information.
    """
    transformed_data = [
        {
            'loc': f"{item['layer']}.{item['sublayer'].split('.')[-1].split('_')[0]}",
            'neuron_index': f"{item['layer']}.{item['sublayer'].split('.')[-1].split('_')[0]}.{item['neuron_index']}"
        }
        for item in data
    ]
    del data  # Free memory
    return transformed_data


# In[ ]:


def compute_similar_neurons(file_list):
    """
    Processes a list of files, counts neuron occurrences for 'white' and 'black' files,
    and prepares results in the desired format.

    Args:
        file_list (list): List of file paths to process.

    Returns:
        df_w (pd.DataFrame): DataFrame of white neuron counts.
        df_b (pd.DataFrame): DataFrame of black neuron counts.
    """
    w_neuron_counts = defaultdict(int)
    b_neuron_counts = defaultdict(int)

    # Process each file in the file list
    for file in tqdm(file_list, desc="Processing files"):
        neuron_list = transform_data(open_pickle(file))
        
        if 'black' in file:
            for item in neuron_list:
                b_neuron_counts[item["neuron_index"]] += 1
        elif 'white' in file:
            for item in neuron_list:
                w_neuron_counts[item["neuron_index"]] += 1
        else:
            raise ValueError('Name in file incompatibility')

    # Prepare the final result format for white neurons
    w_final_result = [{'loc': neuron_index.split('.')[0] + '.' + neuron_index.split('.')[1],
                     'neuron_index': neuron_index,
                     'num_prompts': count} 
                    for neuron_index, count in tqdm(w_neuron_counts.items(), desc="Processing white neurons")]

    # Prepare the final result format for black neurons
    b_final_result = [{'loc': neuron_index.split('.')[0] + '.' + neuron_index.split('.')[1],
                     'neuron_index': neuron_index,
                     'num_prompts': count} 
                    for neuron_index, count in tqdm(b_neuron_counts.items(), desc="Processing black neurons")]

    # Convert lists to DataFrames
    df_w = pd.DataFrame(w_final_result)
    df_b = pd.DataFrame(b_final_result)

    # Split 'loc' into 'layer_num' and 'sublayer' and convert 'layer_num' to numeric
    df_w[['layer_num', 'sublayer']] = df_w['loc'].str.split('.', expand=True)
    df_w['layer_num'] = pd.to_numeric(df_w['layer_num'], downcast='unsigned')

    df_b[['layer_num', 'sublayer']] = df_b['loc'].str.split('.', expand=True)
    df_b['layer_num'] = pd.to_numeric(df_b['layer_num'], downcast='unsigned')

    # Save the DataFrames to CSV files
    df_w.to_csv('top15_w_neurons_12_items_training.csv', index=False)
    df_b.to_csv('top15_b_neurons_12_items_training.csv', index=False)

    return df_w, df_b


# In[ ]:




