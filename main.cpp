#include <iostream>
#include <random>

#include "neural.hpp"

int main(){

    neural_net my_net ;     // defualt initialization of the class

    // adds a layer with 784 ( 28 * 28 ) neurons
    my_net.add_input_layer( 2 ) ;   

    my_net.add_hidden_layer( 2 ) ;

    my_net.add_output_layer( 1 ) ;    // adds output layer

    // creates the networks weights
    
    // creates weights
    my_net.compile() ;

    // displays interesting info
    my_net.summary() ;

    my_net.layers[0]->neurons[0]->value = 0.666 ;
    
    
    // calculates values all the way from start to finish
    my_net.feedforward() ;  

    std::cout << "output : " << my_net.layers[2]->neurons[0]->value << "\n";

    return 0;
}