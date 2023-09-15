# Custom-LSTM-Network
An LSTM network built from scratch using PyTorch. Trained using a spam dataset resulting in a 87.6% accuracy in the validation set. Data was tokenized an split into training and validation sets using 75/25 split.
## Key Formulas
$x_t$ is the sequence at current timestep
$W$ and $U$ are the respective weights of the input and recurrent connections for each gate
$b$ is the respective bias for each gate

- Gates
	- Forget 
		- $\sigma (x_t \times W_f + h_x \times U_f + b_f)$
	- Input 
		- $\sigma (x_t \times W_i + h_x \times U_i + b_i)$
	- Cell 
		- $\tanh(x_t \times W_c + h_x \times U_c + b_c)$
	- Output 
		- $\sigma (x_t \times W_o + h_x \times U_o + b_o)$
- Updating states
	- Cell state
		- $forgetGate \times c_x + inputGate \times cellGate$
	- Hidden State
		- $outputGate \times \tanh(cellState)$
## Customizable Features
- Embedding Dimension
- Number of Hidden Nodes
- Number of Output Nodes
- Number of Layers 
- Dropout Probability
- Creating the network as Bi-directional
