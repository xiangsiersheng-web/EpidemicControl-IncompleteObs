# Predict the first two features (current, new), without using action as features
# Node features are: (current, new)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

from environment.uncertain_seir_vector_v4 import EpidemicModel


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3, bias=True):
        super(GraphConvolution, self).__init__()
        # MLP part: single layer linear transformation
        self.mlp = nn.Linear(in_features, out_features, bias=bias)
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, OD, POP=None, obs_mask=None):
        """
        Forward propagation
        Args:
            x: Node feature matrix, shape (batch_size, num_nodes, in_features)
            OD: Row-normalized adjacency matrix, shape (num_nodes, num_nodes)
            POP: Population data, shape (num_nodes,)
            obs_mask: Unobserved regions in x, shape (batch_size, num_nodes)
        Returns:
            Aggregated node features, shape (batch_size, num_nodes, out_features)
        Note:
            Current weighting assumes x is not divided by POP
        """
        OD = OD.unsqueeze(0).expand(x.shape[0], -1, -1)
        OD_t = OD.transpose(1, 2)

        # # Method 1: MLP first then aggregate
        # x = self.mlp(x)
        # x = F.relu(x)
        # adj = adj.unsqueeze(0).expand(x.shape[0], -1, -1)
        # adj_t = adj.transpose(1, 2)
        # # torch.bmm expects input (batch_size, m, n) @ (batch_size, n, p) -> (batch_size, m, p)
        # x = torch.bmm(adj_t, x)
        # x = torch.bmm(adj, x)
        # return x

        # Use general aggregation directly:
        # POP = None
        # Method 2: Aggregate first then MLP
        if POP is None:
            # # TODO: Switch to standard GCN
            # adj = OD[0] + torch.eye(OD[0].size(0)).to(OD.device)  # Add self-loops
            # deg = adj.sum(dim=1).pow(-0.5)
            # norm_adj = deg.unsqueeze(1) * adj * deg.unsqueeze(0)  # Symmetric normalization
            # norm_adj = norm_adj.unsqueeze(0).expand(x.shape[0], -1, -1)
            #
            # # Feature transformation → Aggregation → Activation
            # x = self.mlp(x)  # Feature transformation
            # x = torch.bmm(norm_adj, x)  # Neighborhood aggregation
            # x = F.relu(x)  # Nonlinear activation
            # return self.dropout(x)

            # Pure weighted matrix aggregation
            # torch.bmm expects input (batch_size, m, n) @ (batch_size, n, p) -> (batch_size, m, p)
            x = torch.bmm(OD, x)
            x = self.mlp(x)
            x = self.dropout(x)

            return F.relu(x)

        else:
            POP = POP.unsqueeze(0).expand(x.shape[0], -1)   # (env.env_count, env.ZONE_NUM)
            predict = torch.bmm(OD_t, x) / torch.bmm(OD_t, POP.unsqueeze(-1))    # (env.env_count, env.ZONE_NUM, 2)
            predict = torch.bmm(OD, predict) * POP.unsqueeze(-1)
            predict = torch.clip(predict, min=0)
            # Numerical correction
            numerical_ratio = [(x[i, ~obs_mask[i], :]).sum(dim=0) / (predict[i, ~obs_mask[i], :].sum(dim=0) + 1e-16)
                               for i in range(x.shape[0])]
            numerical_ratio = torch.stack(numerical_ratio).unsqueeze(1) # (env.env_count, 1, node_features)
            predict = predict * numerical_ratio
            # x[obs_mask] = predict[obs_mask]
            x[obs_mask] = torch.max(predict[obs_mask], x[obs_mask])

            x = self.mlp(x)
            # Apply Dropout
            x = self.dropout(x)

            return F.relu(x)



class RebuildGruGNNModel(nn.Module):
    """GRU-GNN model, inheriting from nn.Module"""

    def __init__(self, gru_hidden_size: object, mlp_output_size: object, gru_num_layers: object, env: object, device_name: object) -> object:
        super(RebuildGruGNNModel, self).__init__()
        self.device = torch.device(device_name)
        self.env = env
        self.zone_num = env.ZONE_NUM
        self.adj = env.OD[0]
        self.POP = env.POP[0] if env.POP is not None else None

        # Define GNN layer
        node_feature_size = 2  #
        node_output_size = 32  # Output node feature dimension
        self.gnn = GraphConvolution(in_features=node_feature_size, out_features=node_output_size, bias=False)

        # Define GRU layer
        self.gru = nn.GRU(
            input_size=node_output_size,  # Dimension of GNN output
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bias=False
        )

        # Define output layer
        # Concatenate GRU hidden_size and zone_num, then input to fully connected layer
        self.fc = nn.Linear(gru_hidden_size, mlp_output_size, bias=False)

    def forward(self, batch_inputs):
        """
        Forward propagation function

        Args:
            batch_inputs (tuple): Input data,
                his_obs: (B, seq_len, zone_num, 2)
                actions: (B, seq_len, zone_num)
                curr_obs: (B, zone_num, 2)

        Returns:
            predictions (torch.Tensor): Prediction results, shape (batch_size, output_size)
        """
        obs, actions = batch_inputs
        batch_size, seq_len, _, _ = obs.shape

        # Store GRU inputs
        gru_inputs = []

        for t in range(seq_len):

            # Extract node features (infection count and actions), shape (batch_size, zone_num, 2)
            i = obs[:, t, :, :]  # (batch_size, zone_num, 2)
            node_features = i  # (batch_size, zone_num, 2)

            # Determine obs_mask from action:
            u_p_test, u_p_quara = self.env._action_to_u(actions[:, t, :])
            obs_mask = u_p_test == 0    # (batch_size, zone_num)

            # Extract features through GNN layer
            gnn_outputs = self.gnn(node_features, self.adj, self.POP, obs_mask)  # (batch_size, zone_num, node_output_size)

            # Use GNN output as GRU input
            gru_inputs.append(gnn_outputs.unsqueeze(1))  # (batch_size, 1, zone_num, node_output_size)


        # Concatenate inputs from all time steps to (batch_size, seq_len + 1, zone_num, node_output_size)
        gru_inputs = torch.cat(gru_inputs, dim=1)

        B, L, M, D = gru_inputs.shape
        gru_inputs = gru_inputs.permute(0, 2, 1, 3).reshape(B*M, L, D)

        # Forward propagate through GRU
        outputs, _ = self.gru(gru_inputs)  # outputs: (B*M, seq_len, hidden_size)

        # Take output from the last time step
        gru_last_output = outputs[:, -1, :]  # (batch_size, hidden_size)

        # Output layer
        predictions = self.fc(gru_last_output)  # (B*M, output_size)
        predictions = predictions.reshape(B, M, -1)

        # Use ReLU activation function
        predictions = F.relu(predictions)

        return predictions


class RebuildGruGNN():
    """Train and evaluate GRU-GNN model"""

    def __init__(self, model, device_name, seq_len):
        self.device = torch.device(device_name)
        self.model = model.to(self.device)
        self.seq_len = seq_len # 7

        self.normalization_stats = {}

        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _calculate_normalization_stats(self, true_state):
        """Calculate normalization parameters"""
        max_curr = torch.max(true_state[..., 0])
        max_new = torch.max(true_state[..., 1])

        print("Normalization parameters - Current: max={:.2f} | New: max={:.2f}".format(max_curr, max_new))

        return {
            'max_curr': max_curr, 'max_new': max_new
        }

    def _normalize(self, x, feature_dim):
        """Normalize"""
        if feature_dim == 0:  # Current cases
            return x / (self.normalization_stats['max_curr'] + 1e-8)
        else:  # New cases
            return x / (self.normalization_stats['max_new'] + 1e-8)

    def _denormalize(self, x, feature_dim):
        """Inverse normalization"""
        if feature_dim == 0:  # Current cases
            return x * (self.normalization_stats['max_curr'] + 1e-8)
        else:  # New cases
            return x * (self.normalization_stats['max_new'] + 1e-8)


    def train(self, dataset, num_epochs=20, batch_size=64, validation_split=0.05, patience=5):
        """Train network using dataset"""
        # Prepare data loader
        inputs, targets = self.prepare_dataset(dataset) # inputs: tuple of (obs, actions)
        dataset = TensorDataset(*inputs, targets)

        # Split training and validation sets
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Evaluate once first
        val_avg_loss, correct_ratio = self.validate(val_loader)
        print(f'Epoch [0], Validation Loss: {val_avg_loss:.6f}, Validation Correct Ratio: {correct_ratio:.4f}')

        # Early stopping parameters
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False
        best_model_state = None

        for epoch in range(num_epochs):
            if early_stop:
                print("Early stopping triggered, stopping training")
                break

            self.model.train()
            epoch_loss = 0.0
            for batch_obs, batch_actions, batch_targets in train_loader:
                batch_obs = batch_obs.to(self.device)  # (batch_size, seq_len, zone_num, 2)
                batch_actions = batch_actions.to(self.device)  # (batch_size, seq_len, zone_num)
                batch_inputs = (batch_obs, batch_actions)
                batch_targets = batch_targets.to(self.device)  # (batch_size, output_size)

                # Forward propagation
                predictions = self.model(batch_inputs)

                # Calculate loss
                loss = self.criterion(predictions, batch_targets)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.6f}')

            # Evaluate model on validation set
            val_avg_loss, correct_ratio = self.validate(val_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_avg_loss:.6f}, Validation Correct Ratio: {correct_ratio:.4f}')

            # Early stopping judgment
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                epochs_no_improve = 0
                best_model_state = self.model.state_dict()  # Save current best model
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Validation loss has not improved for {patience} epochs, stopping training.")
                    early_stop = True

            # After training, load the best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Loaded the best performing model on validation set.")

    def validate(self, val_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch_obs, batch_actions, batch_targets in val_loader:
                batch_obs = batch_obs.to(self.device)  # (batch_size, seq_len, zone_num)
                batch_actions = batch_actions.to(self.device)  # (batch_size, seq_len, zone_num)
                batch_inputs = (batch_obs, batch_actions)
                batch_targets = batch_targets.to(self.device)  # (batch_size, output_size)

                predictions = self.model(batch_inputs)
                loss = self.criterion(predictions, batch_targets)
                total_loss += loss.item()

                # Calculate relative error
                abs_error = torch.abs(predictions - batch_targets)  # (batch_size, zone_num)

                # Handle case where actual value is 0
                relative_error = torch.where(
                    batch_targets != 0,
                    abs_error / torch.abs(batch_targets),
                    torch.where(predictions == 0, torch.zeros_like(abs_error),
                                torch.ones_like(abs_error) * float('inf'))
                )

                # Determine if relative error is less than 30%
                is_correct = relative_error < 0.3  # (batch_size, zone_num)

                # Count correct predictions
                correct_predictions += is_correct.sum().item()
                total_samples += batch_targets.numel()  # batch_size * zone_num

        avg_loss = total_loss / len(val_loader)
        accuracy_rate = (correct_predictions / total_samples) * 100  # percentage
        return avg_loss, accuracy_rate

    def predict(self, obs, actions):
        """Given input, output prediction
        :param obs: Observation, shape=(batch_size, seq_len, zone_num, 2)
        :param actions: Actions, shape=(batch_size, seq_len, zone_num)
        """
        self.model.eval()
        obs = obs.clone()

        obs0 = obs[:, :, :, :2]  # (batch_size, seq_len, zone_num, 2)
        obs0[:, :, :, 0] = self._normalize(obs0[:, :, :, 0], 0)
        obs0[:, :, :, 1] = self._normalize(obs0[:, :, :, 1], 1)

        with torch.no_grad():
            # Preprocess input data
            inputs = (obs0, actions)

            # Forward propagation
            predictions = self.model(inputs)

            # Inverse normalization
            predictions[:, :, 0] = self._denormalize(predictions[:, :, 0], 0)
            predictions[:, :, 1] = self._denormalize(predictions[:, :, 1], 1)

            # Ensure predictions are non-negative
            predictions = torch.clamp(predictions, min=0)

        return predictions  # Return prediction results

    def load(self, model_path):
        """Load model file"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load normalization parameters
        if 'normalization_stats' in checkpoint:
            self.normalization_stats = checkpoint['normalization_stats']
            print("Successfully loaded normalization parameters:", model_path)
        else:
            print("Warning: Normalization parameters not found, prediction results may be inaccurate!")

    def save(self, model_path):
        """Save model file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'normalization_stats': self.normalization_stats
        }, model_path)

    def prepare_dataset(self, dataset):
        """Prepare input and target tensors for training dataset using sliding window approach"""
        obs = dataset['imperfect_obs'][:,:,:,:2]        # (num_samples, total_days, zone_num, 2)
        actions = dataset['history_action']   # (num_samples, total_days, zone_num)
        true_state = dataset['true_state'][:,:,:,:2]    # (num_samples, total_days, zone_num, 2)
        print("obs.shape:", obs.shape, "\t actions.shape:", actions.shape, "\t true_state.shape:", true_state.shape)

        # # TODO: Apply min-max normalization to obs and true_state
        # Calculate and save normalization parameters (based on true data)
        self.normalization_stats = self._calculate_normalization_stats(true_state)

        num_samples, total_days, zone_num, obs_dim = obs.shape

        # Pad seq_len-1 all-zero observations and actions at the front of each data
        obs = torch.cat([torch.zeros((num_samples, self.seq_len - 1, zone_num, obs_dim), device=obs.device), obs], dim=1)
        actions = torch.cat([torch.zeros((num_samples, self.seq_len - 1, zone_num), device=obs.device), actions], dim=1)
        true_state = torch.cat([torch.zeros((num_samples, self.seq_len - 1, zone_num, obs_dim), device=obs.device), true_state], dim=1)

        # Apply normalization
        true_state[:, :, :, 0] = self._normalize(true_state[:, :, :, 0], 0)
        true_state[:, :, :, 1] = self._normalize(true_state[:, :, :, 1], 1)
        obs[:, :, :, 0] = self._normalize(obs[:, :, :, 0], 0)
        obs[:, :, :, 1] = self._normalize(obs[:, :, :, 1], 1)

        num_samples, total_days, zone_num, obs_dim = obs.shape

        inputs_obs = []
        inputs_actions = []
        targets = []

        for i in range(num_samples):
            for day in range(self.seq_len, total_days):  # Note index adjustment here, skip day 0
                # Process observation data
                obs_seq = obs[i, day - self.seq_len + 1 : day + 1, :]  # (seq_len, zone_num, 2)

                # Process action data (based on understanding of s_t, a_t, s_t+1, action affecting s_t+1 is stored on day t
                action_seq = actions[i, day - self.seq_len : day, :]  # (seq_len, zone_num)

                # Get true state of current day
                target = true_state[i, day, :]  # (zone_num, 2)

                inputs_obs.append(obs_seq)
                inputs_actions.append(action_seq)
                targets.append(target)

        # Convert lists to tensors
        inputs_obs = torch.stack(inputs_obs)                # (total_samples, seq_len, zone_num, 2)
        inputs_actions = torch.stack(inputs_actions)        # (total_samples, seq_len, zone_num)
        targets = torch.stack(targets)                      # (total_samples, zone_num, 2)
        print("inputs_obs.shape:", inputs_obs.shape, "\t inputs_actions.shape:", inputs_actions.shape, "\t targets.shape:", targets.shape)

        return (inputs_obs, inputs_actions), targets
