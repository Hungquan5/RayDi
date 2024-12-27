import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = True
        self.train = True
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weights = []
        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size * 2
            
            # Forward weights
            layer_weights = {
                'Wxh': np.random.randn(self.hidden_size, layer_input_size) * 0.01,
                'Whh': np.random.randn(self.hidden_size, self.hidden_size) * 0.01,
                'bh': np.zeros((self.hidden_size, 1)),
                'Wxh_grad': np.zeros((self.hidden_size, layer_input_size)),
                'Whh_grad': np.zeros((self.hidden_size, self.hidden_size)),
                'bh_grad': np.zeros((self.hidden_size, 1))
            }
            
            # Backward weights
            layer_weights_back = {
                'Wxh': np.random.randn(self.hidden_size, layer_input_size) * 0.01,
                'Whh': np.random.randn(self.hidden_size, self.hidden_size) * 0.01,
                'bh': np.zeros((self.hidden_size, 1)),
                'Wxh_grad': np.zeros((self.hidden_size, layer_input_size)),
                'Whh_grad': np.zeros((self.hidden_size, self.hidden_size)),
                'bh_grad': np.zeros((self.hidden_size, 1))
            }
            
            self.weights.append((layer_weights, layer_weights_back))
        
        # Output layer weights
        self.Wy = np.random.randn(self.output_size, self.hidden_size * 2) * 0.01
        self.by = np.zeros((self.output_size, 1))
        self.Wy_grad = np.zeros((self.output_size, self.hidden_size * 2))
        self.by_grad = np.zeros((self.output_size, 1))
    
    def rnn_step_forward(self, x, prev_h, Wxh, Whh, bh):
        h_next = np.tanh(np.dot(Wxh, x) + np.dot(Whh, prev_h) + bh)
        cache = (x, prev_h, Wxh, Whh, h_next)
        return h_next, cache
    
    def rnn_step_backward(self, dh_next, cache):
        x, prev_h, Wxh, Whh, h_next = cache
        dtanh = dh_next * (1 - h_next ** 2)
        dx = np.dot(Wxh.T, dtanh)
        dWxh = np.dot(dtanh, x.T)
        dWhh = np.dot(dtanh, prev_h.T)
        dprev_h = np.dot(Whh.T, dtanh)
        dbh = np.sum(dtanh, axis=1, keepdims=True)
        return dx, dprev_h, dWxh, dWhh, dbh
    
    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        layer_outputs = []
        
        for layer in range(self.num_layers):
            h_forward = np.zeros((batch_size, seq_len, self.hidden_size))
            h_backward = np.zeros((batch_size, seq_len, self.hidden_size))
            
            # Forward pass
            h_prev = np.zeros((self.hidden_size, batch_size))
            forward_caches = []
            
            for t in range(seq_len):
                x_t = X[:, t, :].T if layer == 0 else layer_outputs[-1][:, t, :].T
                h_next, cache = self.rnn_step_forward(
                    x_t, h_prev,
                    self.weights[layer][0]['Wxh'],
                    self.weights[layer][0]['Whh'],
                    self.weights[layer][0]['bh']
                )
                h_forward[:, t, :] = h_next.T
                forward_caches.append(cache)
                h_prev = h_next
            
            # Backward pass
            h_prev = np.zeros((self.hidden_size, batch_size))
            backward_caches = []
            
            for t in range(seq_len - 1, -1, -1):
                x_t = X[:, t, :].T if layer == 0 else layer_outputs[-1][:, t, :].T
                h_next, cache = self.rnn_step_forward(
                    x_t, h_prev,
                    self.weights[layer][1]['Wxh'],
                    self.weights[layer][1]['Whh'],
                    self.weights[layer][1]['bh']
                )
                h_backward[:, t, :] = h_next.T
                backward_caches.append(cache)
                h_prev = h_next
            
            # Concatenate directions
            h_concat = np.concatenate([h_forward, h_backward], axis=2)
            
            # Apply dropout
            if self.train and self.dropout > 0:
                mask = (np.random.rand(*h_concat.shape) > self.dropout) / (1 - self.dropout)
                h_concat *= mask
            
            layer_outputs.append(h_concat)
        
        # Final output
        last_hidden = layer_outputs[-1][:, -1, :]
        output = np.dot(last_hidden, self.Wy.T) + self.by.T
        
        self.caches = {
            'layer_outputs': layer_outputs,
            'forward_caches': forward_caches,
            'backward_caches': backward_caches,
            'last_hidden': last_hidden
        }
        
        return output
    
    def backward(self, dout):
        batch_size = dout.shape[0]
        
        # Output layer gradients
        self.Wy_grad = np.dot(dout.T, self.caches['last_hidden'])
        self.by_grad = np.sum(dout.T, axis=1, keepdims=True)
        
        dlast_hidden = np.dot(dout, self.Wy)
        
        for layer in reversed(range(self.num_layers)):
            dh = dlast_hidden
            
            # Backward through time
            dh_prev = np.zeros((self.hidden_size, batch_size))
            for t in reversed(range(len(self.caches['forward_caches']))):
                dx, dh_prev, dWxh, dWhh, dbh = self.rnn_step_backward(
                    dh[:, t:t+1], self.caches['forward_caches'][t]
                )
                self.weights[layer][0]['Wxh_grad'] += dWxh
                self.weights[layer][0]['Whh_grad'] += dWhh
                self.weights[layer][0]['bh_grad'] += dbh
                
                if layer > 0:
                    dlast_hidden = dx
    
    def save(self, filepath):
        weights_list = []
        for layer_weights in self.weights:
            weights_list.append({
                'forward': {k: v for k, v in layer_weights[0].items() if not k.endswith('_grad')},
                'backward': {k: v for k, v in layer_weights[1].items() if not k.endswith('_grad')}
            })
        
        np.save(filepath, {
            'weights': weights_list,
            'Wy': self.Wy,
            'by': self.by
        })
    
    def load(self, filepath):
        data = np.load(filepath, allow_pickle=True).item()
        
        for layer, weights in enumerate(data['weights']):
            for k, v in weights['forward'].items():
                self.weights[layer][0][k] = v
            for k, v in weights['backward'].items():
                self.weights[layer][1][k] = v
        
        self.Wy = data['Wy']
        self.by = data['by']