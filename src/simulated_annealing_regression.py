#!/usr/bin/env python
# coding: utf-8

# In[1]:

#pip install tensorflow



# In[10]:


#pip install xeus-python


# In[1]:


import tensorflow as tf
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import pdb
import math


# In[4]:


class NeuronalNetworkRegression:
    def __init__(self, activation_function, num_layers, nodes_per_layer, input_shape, output_shape):
        #Initialisierung eines sequentiellen Modells (an Schichten)
        self.model = tf.keras.Sequential()
        
        #Hinzufügen der Eingangsschicht zum NN (=Anzahl an Input-Neuronen)
        self.model.add(tf.keras.layers.InputLayer(input_shape))

        #Hinzufügen versteckter Schichten
        for _ in range(int(num_layers)):
            self.model.add(tf.keras.layers.Dense(nodes_per_layer, activation=activation_function))
            ##Verbindung jedes Neurons mit jedem aus der vorherigen (durch .Dense)

        #Ausgangsschicht
        self.model.add(tf.keras.layers.Dense(output_shape))  
            ##'softmax' für Klassifikation (n Ausgangsschichten für jede Kategorie notwendig --> softmax liefert p(x) für jede Kategorie)
        
        #Konfiguration des Modells während des Trainings
        self.model.compile(optimizer='adam', loss='mean_squared_error')
            ##optimizer ... Aktualisierung der Gewichtung während des Trainings
            ##loss ... Quantifizierung zwischen Vorhersage und Tatsache 
            ##metrics ... Bewertungsmatrix während des Trainings

    def train(self, train_data, train_labels, epochs=10, batch_size=32):
        self.model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
        ##epochs ... Anzahl der Iterationen über den Trainingsdatensatz
        ##batch_size ... Anzahl der Verwendung an Datenpunkte pro Schritt für Gewichtsaktualisierung

    def evaluate(self, validation_data, validation_labels):
        loss = self.model.evaluate(validation_data, validation_labels)
        return loss  
        ##loss ... Abweichung zw. Vorhersage und Tatsache

#NN
##Hyperparameter: Hidden-Layer, Aktivitätsfunktion
activation_functions = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
layer_ranges = list(range(1, 11))
node_ranges = list(range(1,121,1))  

bounds = np.asarray([[0, len(activation_functions)-1], [1, 10], [1, 120]])

n_iterations = 20
temp = 10
step_size = 1


# In[ ]:


# simulated annealing algorithm
def simulated_annealing(neural_network_class, bounds, activation_functions, train_data, train_labels, 
                        validation_data, validation_labels, n_iterations, step_size, temp, input_shape, output_shape):
    # generate an initial point
    best_activation = np.random.randint(bounds[0, 0], bounds[0, 1])
    best_activation_fct = activation_functions[best_activation]
    
    best_layer = np.random.randint(bounds[1, 0], bounds[1, 1])
    best_nodes = np.random.randint(bounds[2, 0], bounds[2, 1])
    
    # evaluate the initial point
    best_nn = neural_network_class(best_activation_fct, best_layer, best_nodes, input_shape, output_shape)
    best_nn.train(train_data, train_labels)
    
    best_eval = best_nn.evaluate(validation_data, validation_labels)
    
    # current working solution
    curr_activation, curr_activation_fct, curr_layer, curr_nodes, curr_eval, curr_nn = best_activation, best_activation_fct, best_layer, best_nodes, best_eval, best_nn
    
    candidate_eval_best = None # for for loop later, so that the old value is safed for further iterations
    candidate_index_best = None 
    # run the algorithm
    for i in range(n_iterations):
        
        # take a step in activation function
        #candidate_activation = curr_activation + np.round(randn() * step_size)
        #if candidate_activation > bounds[0, 1] + 1:
            #candidate_activation -= bounds[0, 1] - 1
            
        #candidate_activation_fct = activation_functions[candidate_activation - 1]
        
        delta_activation = int(np.round(np.random.rand() * step_size))
        candidate_activation_forw = curr_activation + delta_activation
        candidate_activation_backw = curr_activation - delta_activation
        
        if candidate_activation_forw > (bounds[0, 1]):
            candidate_activation_forw -= (bounds[0, 1] + 1)
        
        elif candidate_activation_forw < (bounds[0, 0]):
            candidate_activation_forw += (bounds[0, 1] + 1)
            
        #print(candidate_activation_forw - 1)    
        #print(activation_functions[candidate_activation_forw - 1])
        #pdb.set_trace()
        candidate_activation_forw_fct = activation_functions[candidate_activation_forw]
        
        if candidate_activation_backw < (bounds[0, 0]):
            candidate_activation_backw += (bounds[0, 1] + 1)
            
        elif candidate_activation_backw > (bounds[0, 1]):
            candidate_activation_backw -= (bounds[0, 1] + 1)
            
        candidate_activation_backw_fct = activation_functions[candidate_activation_backw]
            
        # take a step in layer    
        #candidate_layer = curr_layer + np.round(randn() * step_size)
        #if candidate_layer > (bounds[1, 1] + 1):
            #candidate_layer -= (bounds[1, 1] + 1)
            
        delta_layer = np.round(np.random.randn() * step_size)
        candidate_layer_forw = curr_layer + delta_layer
        candidate_layer_backw = curr_layer - delta_layer
        
        if candidate_layer_forw > (bounds[1, 1]):
            candidate_layer_forw -= (bounds[1, 1] + 1)
            
        if candidate_layer_backw < (bounds[1, 0]):
            candidate_layer_backw += (bounds[1, 1] + 1)    
        
        # take a step in nodes
        #candidate_nodes = curr_nodes + np.round(randn() * step_size)
        #if candidate_nodes > (bounds[2, 1] + 1):
            #candidate_nodes -= (bounds[2, 1] + 1)
            
        delta_nodes = np.round(np.random.randn() * step_size)
        candidate_nodes_forw = curr_nodes + delta_nodes
        candidate_nodes_backw = curr_nodes - delta_nodes
        
        if candidate_nodes_forw > (bounds[2, 1]):
            candidate_nodes_forw -= (bounds[2, 1] + 1)
            
        if candidate_nodes_backw < (bounds[2, 0]):
            candidate_nodes_backw += (bounds[2, 1] + 1)   
        
        # evaluate candidate point
        ##legend
        ### x ... step forward
        ### y ... step backward
        ### 0 ... no step
        
        ###epsilon environment in a three-dimensional parameterspace
        
        ##steps forward ---------------------------------------------------------------------------
        candidate_x00 = neural_network_class(candidate_activation_forw_fct, curr_layer, curr_nodes, 
                                              input_shape, output_shape)
        candidate_x00.train(train_data, train_labels)
        candidate_eval_x00 = candidate_x00.evaluate(validation_data, validation_labels)
        
        
        candidate_0x0 = neural_network_class(curr_activation_fct, candidate_layer_forw, curr_nodes, 
                                              input_shape, output_shape)
        candidate_0x0.train(train_data, train_labels)
        candidate_eval_0x0 = candidate_0x0.evaluate(validation_data, validation_labels)
        
        
        candidate_00x = neural_network_class(curr_activation_fct, curr_layer, candidate_nodes_forw, 
                                              input_shape, output_shape)
        candidate_00x.train(train_data, train_labels)
        candidate_eval_00x = candidate_00x.evaluate(validation_data, validation_labels)
        
        
        candidate_xx0 = neural_network_class(candidate_activation_forw_fct, candidate_layer_forw, curr_nodes, 
                                              input_shape, output_shape)
        candidate_xx0.train(train_data, train_labels)
        candidate_eval_xx0 = candidate_xx0.evaluate(validation_data, validation_labels)
        
        
        candidate_x0x = neural_network_class(candidate_activation_forw_fct, curr_layer, candidate_nodes_forw, 
                                              input_shape, output_shape)
        candidate_x0x.train(train_data, train_labels)
        candidate_eval_x0x = candidate_x0x.evaluate(validation_data, validation_labels)
        
        
        candidate_0xx = neural_network_class(curr_activation_fct, candidate_layer_forw, candidate_nodes_forw, 
                                              input_shape, output_shape)
        candidate_0xx.train(train_data, train_labels)
        candidate_eval_0xx = candidate_0xx.evaluate(validation_data, validation_labels)
        
        
        candidate_xxx = neural_network_class(candidate_activation_forw_fct, candidate_layer_forw, candidate_nodes_forw, 
                                              input_shape, output_shape)
        candidate_xxx.train(train_data, train_labels)
        candidate_eval_xxx = candidate_xxx.evaluate(validation_data, validation_labels)
            
        
        ##steps backward -----------------------------------------------------------------------
        candidate_y00 = neural_network_class(candidate_activation_backw_fct, curr_layer, curr_nodes, 
                                              input_shape, output_shape)
        candidate_y00.train(train_data, train_labels)
        candidate_eval_y00 = candidate_y00.evaluate(validation_data, validation_labels)
        
        
        candidate_0y0 = neural_network_class(curr_activation_fct, candidate_layer_backw, curr_nodes, 
                                              input_shape, output_shape)
        candidate_0y0.train(train_data, train_labels)
        candidate_eval_0y0 = candidate_0y0.evaluate(validation_data, validation_labels)
        
        
        candidate_00y = neural_network_class(curr_activation_fct, curr_layer, candidate_nodes_backw, 
                                              input_shape, output_shape)
        candidate_00y.train(train_data, train_labels)
        candidate_eval_00y = candidate_00y.evaluate(validation_data, validation_labels)
        
        
        candidate_yy0 = neural_network_class(candidate_activation_backw_fct, candidate_layer_backw, curr_nodes, 
                                              input_shape, output_shape)
        candidate_yy0.train(train_data, train_labels)
        candidate_eval_yy0 = candidate_yy0.evaluate(validation_data, validation_labels)
        
        
        candidate_y0y = neural_network_class(candidate_activation_backw_fct, curr_layer, candidate_nodes_backw, 
                                              input_shape, output_shape)
        candidate_y0y.train(train_data, train_labels)
        candidate_eval_y0y = candidate_y0y.evaluate(validation_data, validation_labels)
        
        
        candidate_0yy = neural_network_class(curr_activation_fct, candidate_layer_backw, candidate_nodes_backw, 
                                              input_shape, output_shape)
        candidate_0yy.train(train_data, train_labels)
        candidate_eval_0yy = candidate_0yy.evaluate(validation_data, validation_labels)
        
        
        candidate_yyy = neural_network_class(candidate_activation_backw_fct, candidate_layer_backw, candidate_nodes_backw, 
                                              input_shape, output_shape)
        candidate_yyy.train(train_data, train_labels)
        candidate_eval_yyy = candidate_yyy.evaluate(validation_data, validation_labels)
        
        
        ##steps forward & backward -------------------------------------------------
        candidate_yxx = neural_network_class(candidate_activation_backw_fct, candidate_layer_forw, candidate_nodes_forw, 
                                              input_shape, output_shape)
        candidate_yxx.train(train_data, train_labels)
        candidate_eval_yxx = candidate_yxx.evaluate(validation_data, validation_labels)
        
        
        candidate_xyx = neural_network_class(candidate_activation_forw_fct, candidate_layer_backw, candidate_nodes_forw, 
                                              input_shape, output_shape)
        candidate_xyx.train(train_data, train_labels)
        candidate_eval_xyx = candidate_xyx.evaluate(validation_data, validation_labels)
        
        
        candidate_xxy = neural_network_class(candidate_activation_forw_fct, candidate_layer_forw, candidate_nodes_backw, 
                                              input_shape, output_shape)
        candidate_xxy.train(train_data, train_labels)
        candidate_eval_xxy = candidate_xxy.evaluate(validation_data, validation_labels)
        
        
        candidate_xyy = neural_network_class(candidate_activation_forw_fct, candidate_layer_backw, candidate_nodes_backw, 
                                              input_shape, output_shape)
        candidate_xyy.train(train_data, train_labels)
        candidate_eval_xyy = candidate_xyy.evaluate(validation_data, validation_labels)
        
        
        candidate_yxy = neural_network_class(candidate_activation_backw_fct, candidate_layer_forw, candidate_nodes_backw, 
                                              input_shape, output_shape)
        candidate_yxy.train(train_data, train_labels)
        candidate_eval_yxy = candidate_yxy.evaluate(validation_data, validation_labels)
        
        
        candidate_yyx = neural_network_class(candidate_activation_backw_fct, candidate_layer_backw, candidate_nodes_forw, 
                                              input_shape, output_shape)
        candidate_yyx.train(train_data, train_labels)
        candidate_eval_yyx = candidate_yyx.evaluate(validation_data, validation_labels)
        
        
        ##steps forward & backward & no steps -------------------------------------------
        candidate_yx0 = neural_network_class(candidate_activation_backw_fct, candidate_layer_forw, curr_nodes, 
                                              input_shape, output_shape)
        candidate_yx0.train(train_data, train_labels)
        candidate_eval_yx0 = candidate_yx0.evaluate(validation_data, validation_labels)
        
        
        candidate_xy0 = neural_network_class(candidate_activation_forw_fct, candidate_layer_backw, curr_nodes, 
                                              input_shape, output_shape)
        candidate_xy0.train(train_data, train_labels)
        candidate_eval_xy0 = candidate_xy0.evaluate(validation_data, validation_labels)
        
        
        candidate_0yx = neural_network_class(curr_activation_fct, candidate_layer_backw, candidate_nodes_forw, 
                                              input_shape, output_shape)
        candidate_0yx.train(train_data, train_labels)
        candidate_eval_0yx = candidate_0yx.evaluate(validation_data, validation_labels)
        
        
        candidate_0xy = neural_network_class(curr_activation_fct, candidate_layer_forw, candidate_nodes_backw, 
                                              input_shape, output_shape)
        candidate_0xy.train(train_data, train_labels)
        candidate_eval_0xy = candidate_0xy.evaluate(validation_data, validation_labels)
        
        
        candidate_y0x = neural_network_class(candidate_activation_backw_fct, curr_layer, candidate_nodes_forw, 
                                              input_shape, output_shape)
        candidate_y0x.train(train_data, train_labels)
        candidate_eval_y0x = candidate_y0x.evaluate(validation_data, validation_labels)
        
        
        candidate_x0y = neural_network_class(candidate_activation_forw_fct, curr_layer, candidate_nodes_backw, 
                                              input_shape, output_shape)
        candidate_x0y.train(train_data, train_labels)
        candidate_eval_x0y = candidate_x0y.evaluate(validation_data, validation_labels)
        #---------------------------------------------------------------------------------------------------------
        
        #dictionary for the current epsilon environment
        candidate = {
                'evaluation': [
                          candidate_eval_x00, candidate_eval_0x0, candidate_eval_00x, candidate_eval_xx0, #step forward
                              candidate_eval_x0x, candidate_eval_0xx, candidate_eval_xxx, 
                    
                          candidate_eval_y00, candidate_eval_0y0, candidate_eval_00y, candidate_eval_yy0, #step backward
                              candidate_eval_y0y, candidate_eval_0yy, candidate_eval_yyy, 
                    
                          candidate_eval_yxx, candidate_eval_xyx, candidate_eval_xxy, #steps forward & backward
                              candidate_eval_yyx, candidate_eval_yxy, candidate_eval_xyy, 
                    
                          candidate_eval_yx0, candidate_eval_xy0, candidate_eval_y0x, #steps forward & backward & no step
                              candidate_eval_0xy, candidate_eval_0yx, candidate_eval_x0y 
                              ], 
            
                'activation': [
                         candidate_activation_forw, curr_activation, curr_activation, candidate_activation_forw, #step forward
                             candidate_activation_forw, curr_activation, candidate_activation_forw, 
                    
                         candidate_activation_backw, curr_activation, curr_activation, candidate_activation_backw, #step backward
                             candidate_activation_backw, curr_activation, candidate_activation_backw, 
                    
                         candidate_activation_backw, candidate_activation_forw, candidate_activation_forw, #steps forward & backward
                             candidate_activation_backw, candidate_activation_backw, candidate_activation_forw, 
                    
                         candidate_activation_backw, candidate_activation_forw, candidate_activation_backw, #steps forward & backward & no step
                             curr_activation, curr_activation, candidate_activation_forw 
                              ], 
            
                'activation_fct': [
                         candidate_activation_forw_fct, curr_activation_fct, curr_activation_fct, candidate_activation_forw_fct, #step forward
                             candidate_activation_forw_fct, curr_activation_fct, candidate_activation_forw_fct, 
                    
                         candidate_activation_backw_fct, curr_activation_fct, curr_activation_fct, candidate_activation_backw_fct, #step backward
                             candidate_activation_backw_fct, curr_activation_fct, candidate_activation_backw_fct, 
                    
                         candidate_activation_backw_fct, candidate_activation_forw_fct, candidate_activation_forw_fct, #steps forward & backward
                             candidate_activation_backw_fct, candidate_activation_backw_fct, candidate_activation_forw_fct,
                    
                         candidate_activation_backw_fct, candidate_activation_forw_fct, candidate_activation_backw_fct, #steps forward & backward & no step
                             curr_activation_fct, curr_activation_fct, candidate_activation_forw_fct 
                                  ],
            
                'layer': [
                         curr_layer, candidate_layer_forw, curr_layer, candidate_layer_forw, #step forward
                             curr_layer, candidate_layer_forw, candidate_layer_forw,
                    
                         curr_layer, candidate_layer_backw, curr_layer, candidate_layer_backw, #step backward
                             curr_layer, candidate_layer_backw, candidate_layer_backw,
                    
                         candidate_layer_forw, candidate_layer_backw, candidate_layer_forw, #steps forward & backward
                             candidate_layer_backw, candidate_layer_forw, candidate_layer_backw,
                    
                         candidate_layer_forw, candidate_layer_backw, curr_layer, #steps forward & backward & no step
                             candidate_layer_forw, candidate_layer_backw, curr_layer,
                         ],
            
                'nodes': [
                        curr_nodes, curr_nodes, candidate_nodes_forw, curr_nodes, #step forward
                          candidate_nodes_forw, candidate_nodes_forw, candidate_nodes_forw,
                    
                        curr_nodes, curr_nodes, candidate_nodes_backw, curr_nodes, #step backward
                          candidate_nodes_backw, candidate_nodes_backw, candidate_nodes_backw,
                    
                        candidate_nodes_forw, candidate_nodes_forw, candidate_nodes_backw, #steps forward & backward
                          candidate_nodes_forw, candidate_nodes_backw, candidate_nodes_backw,
                    
                        curr_nodes, curr_nodes, candidate_nodes_forw,  #steps forward & backward & no step
                          candidate_nodes_backw, candidate_nodes_forw, candidate_nodes_backw,
                         ],
                'NN': [
                        candidate_x00, candidate_0x0, candidate_00x, candidate_xx0, #step forward
                            candidate_x0x, candidate_0xx, candidate_xxx, 
                  
                        candidate_y00, candidate_0y0, candidate_00y, candidate_yy0, #step backward
                            candidate_y0y, candidate_0yy, candidate_yyy, 
                  
                        candidate_yxx, candidate_xyx, candidate_xxy, #steps forward & backward
                            candidate_yyx, candidate_yxy, candidate_xyy, 
                  
                        candidate_yx0, candidate_xy0, candidate_y0x, #steps forward & backward & no step
                            candidate_0xy, candidate_0yx, candidate_x0y 
                      ]
            }
        
        counter = 0
        # check for new best solution
        for index, element in enumerate(candidate['evaluation']):
            if element < best_eval:
                if candidate_eval_best is None or element < candidate_eval_best:
                    candidate_eval_best = element
                    candidate_index_best = index
        
                    # store new best point
                    best_activation, best_activation_fct, best_layer, best_nodes, best_eval, best_nn = candidate['activation'][candidate_index_best], candidate['activation_fct'][candidate_index_best], candidate['layer'][candidate_index_best], candidate['nodes'][candidate_index_best], candidate_eval_best, candidate['NN'][candidate_index_best]
                    
                    # report progress
                    print('>%d (fct_id = %d and number of layers = %d and number of nodes = %d) f(%s) = %.5f' % (element, best_activation, best_layer, best_nodes, best_activation_fct, best_eval))
            elif element > best_eval:
                counter += 1 # count the times that we did not find a better candidate
    
        if counter == len(candidate['evaluation']): #if we havent found any better solution then we assign the highest value of the evaltuation in this iteration to as the pseudo candidate
            candidate_eval_best = max(candidate['evaluation'])
        
        # difference between candidate and current point evaluation
        diff = curr_eval - candidate_eval_best #diff < 0 for better evaluations
        print(diff)
        if diff < -1: #for normalizing high differences not to conflict with exp(-diff / t)
            diff = -1
        
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        
        # calculate metropolis acceptance criterion
        metropolis = math.exp(-diff / t) #if diff < 0 (the solution is actually not better) then metropolis follows an exponential decay over time (iterations)
        
        # check if we should keep the new point
        if diff < 0 or np.random.rand() < metropolis: #either the solution is better or we allow a worse solution (tolerance decreases by time)
        # store the new current point
            curr_activation, curr_activation_fct, curr_layer, curr_nodes, curr_eval, curr_nn = candidate['activation'][candidate_index_best], candidate['activation_fct'][candidate_index_best], candidate['layer'][candidate_index_best], candidate['nodes'][candidate_index_best], candidate_eval_best, candidate['NN'][candidate_index_best]
            
    return [best_activation_fct, best_layer, best_nodes, best_eval, best_nn]
    #return [curr_activation_fct, curr_layer, curr_nodes, curr_eval] #???


# In[7]:


pfad = 'C:/Users/benja/Documents/Persoenlich/Ausbildung/TU/Digitale_Kompetenzen/03_Anwendungsfelder_der_Digitalisierung/Machine_Learning/UE/2_Exercise/daten/'
df = pd.read_csv(pfad + 'buildings.csv')

target_col = 'y'
feature_cols = df.columns.difference([target_col])

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(df[feature_cols], df[target_col], test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert the pandas DataFrame and Series into TensorFlow Tensors
#X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
#X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
#X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)

# For the target variable, ensure it's encoded properly if it's categorical
# Here, I'm assuming 'grade' is categorical. You might need to adjust this part based on your data.
#y_train_tensor = tf.convert_to_tensor(pd.get_dummies(y_train), dtype=tf.float32)
#y_val_tensor = tf.convert_to_tensor(pd.get_dummies(y_val), dtype=tf.float32)
#y_test_tensor = tf.convert_to_tensor(pd.get_dummies(y_test), dtype=tf.float32)

X_train_tensor = X_train
X_val_tensor =X_val
X_test_tensor =X_test

y_train_tensor =y_train
y_val_tensor = y_val
y_test_tensor =y_test

print(X_train)
print(X_train_tensor)
print("  --------------------")

print(y_train)
print(y_train_tensor)


X_train.shape[1]


# In[ ]:


# Run the simulated annealing
best_activation_fct, best_layer, best_nodes, best_eval, best_nn = simulated_annealing(NeuronalNetworkRegression, bounds, activation_functions, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, n_iterations, step_size, temp, input_shape = X_train.shape[1], output_shape = 1)

#Output the best configuration found
print(f"Best Activation Fct: {best_activation_fct}, Best Number of Layers: {best_layer}, Best Number of Nodes per Layer: {best_nodes}, Performance: {best_eval}")


# In[ ]:

y_pred = best_nn.model.predict(X_test_tensor)

import matplotlib.pyplot as plt
import numpy as np

# Assuming you have trained your regression model and obtained predictions
y_true = y_test_tensor  # Replace with your true labels (ground truth)
  # Replace with your predicted labels


# Create a scatter plot of true vs. predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.title('True vs. Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.grid(True)

# Add a diagonal line for reference (perfect predictions)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')

# Show the plot
plt.show()


