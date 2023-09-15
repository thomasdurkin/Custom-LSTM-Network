# Custom-LSTM-Network
An LSTM network built from scratch using PyTorch. Trained using a spam dataset resulting in a 87.6% accuracy in the validation set. Data was tokenized an split into training and validation sets using 75/25 split.

## Sample Data
| type  | text                                                                                                                                                           |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ham   | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...                                                |
| ham   | I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.                                                  |
| spam  | URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18    |
| ham   | I‘m going to try for 2 months ha ha only joking                                                                                                                |
| spam  | Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO. If you reply NO you will not be charged |

## Key Formulas
$x_t$ is the sequence at current timestep</br>
$W$ and $U$ are the respective weights of the input and recurrent connections for each gate</br>
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
