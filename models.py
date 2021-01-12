# This class defines the architecture for model with bottleneck = 2 

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_1 = nn.Linear(in_features = 11, out_features = 82)
        self.encoder_2 = nn.Linear(in_features = 82, out_features = 80)
        self.encoder_3 = nn.Linear(in_features = 80, out_features = 77)
        self.encoder_4 = nn.Linear(in_features = 77, out_features = 29)
        self.encoder_5 = nn.Linear(in_features = 29, out_features = 2)
        
        self.decoder_5 = nn.Linear(in_features = 2, out_features = 29)
        self.decoder_4 = nn.Linear(in_features = 29, out_features = 77)
        self.decoder_3 = nn.Linear(in_features = 77, out_features = 80)
        self.decoder_2 = nn.Linear(in_features = 80, out_features = 82)
        self.decoder_1 = nn.Linear(in_features = 82, out_features = 11)
            
    def forward(self, features):
        LeakyRelu = nn.LeakyReLU(0.2)
        # Encoder 
        x1         = self.encoder_1(features)
        activation = LeakyRelu(x1)
        x2         = self.encoder_2(activation)
        activation = LeakyRelu(x2)
        x3         = self.encoder_3(activation)
        activation = LeakyRelu(x3)
        x4         = self.encoder_4(activation)
        activation = LeakyRelu(x4)
        x5         = self.encoder_5(activation)
        activation = LeakyRelu(x5)
        
        # Decoder
        x6         = self.decoder_5(x5)
        activation = LeakyRelu(x6)
        x7         = self.decoder_4(activation)
        activation = LeakyRelu(x7)
        x8         = self.decoder_3(activation)
        activation = LeakyRelu(x8)
        x9         = self.decoder_2(activation)
        activation = LeakyRelu(x9)
        x10        = self.decoder_1(activation)
        
        # Return bottleneck 
        return x5
        
# This is another class that defines the architecture for model with bottleneck = 2 
class AE(nn.Module):
    def __init__(self, input_size, bottleneck, out_features, n_layers):
        super().__init__()
        self.input_size = input_size
        self.bottleneck = bottleneck
        self.out_features = out_features
        self.n_layers = n_layers
        
        encoder_layers = []
        decoder_layers = []
        
        for i in range(n_layers):
            if i == 0: # First and last layers of the model
                if i == n_layers - 1:  # if only 1 hidden layer
                    # Add encoder input layer 
                    encoder_layers.append(nn.Linear(input_size, bottleneck_neurons))
                    encoder_layers.append(nn.LeakyReLU(0.2))

                    # Add final decoder output layer
                    decoder_layers.append(nn.Linear(bottleneck_neurons, input_size))
                    # No activation layer here (decoder output)

                else: 
                    # Add encoder input layer 
                    encoder_layers.append(nn.Linear(input_size, out_features[0]))
                    encoder_layers.append(nn.LeakyReLU(0.2))

                    # Define in_features to be out_features for decoder
                    in_features = out_features[0]

                    # Add final decoder output layer
                    decoder_layers.append(nn.Linear(in_features, input_size))
                    # No activation layer here (decoder output)

            elif i == n_layers - 1: 
                # add the layers adjacent to the bottleneck
                encoder_layers.append(nn.Linear(in_features, bottleneck))
                encoder_layers.append(nn.LeakyReLU(0.2))

                decoder_layers.append(nn.LeakyReLU(0.2))
                decoder_layers.append(nn.Linear(bottleneck, in_features))

            else:
                # Add encoder layers
                encoder_layers.append(nn.Linear(in_features, out_features[i]))
                encoder_layers.append(nn.LeakyReLU(0.2))

                # Define in_features to be out_features for decoder
                in_features = out_features[i] 

                # Add decoder layers
                decoder_layers.append(nn.LeakyReLU(0.2))
                decoder_layers.append(nn.Linear(in_features, out_features[i-1]))
    
        # Reverse order of layers in decoder list
        decoder_layers.reverse()

        # return the parts of model
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.decoder_layers = nn.Sequential(*decoder_layers)
    
    def forward(self, features):
        x1 = self.encoder_layers(features)
        x2 = self.decoder_layers(x1)
        return x1
