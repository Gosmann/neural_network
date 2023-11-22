#include <iostream>
#include <random>
#include <chrono>
#include <thread>

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

    //my_net.layers[0]->neurons[0]->value = 0.666 ;
    //my_net.layers[0]->neurons[0]->feedforward() ; 
    
    // displays interesting info
    my_net.summary() ;

    // calculates values all the way from start to finish
    my_net.feedforward() ;  

    std::cout << "layer 1 neuron 0 : \n";
    std::cout << "output value : " << my_net.layers[1]->neurons[0]->value << "\n";
    std::cout << "output activated: " << my_net.layers[1]->neurons[0]->activated << "\n";

    std::cout << "layer 1 neuron 1 : \n";
    std::cout << "output value : " << my_net.layers[1]->neurons[1]->value << "\n";
    std::cout << "output activated: " << my_net.layers[1]->neurons[1]->activated << "\n";

    std::cout << "layer 2 neuron 0 : \n";
    std::cout << "output value : " << my_net.layers.back()->neurons[0]->value << "\n";
    std::cout << "output activated: " << my_net.layers.back()->neurons[0]->activated << "\n";

    layer input ( 2, layer::input ) ;
    input.neurons[0]->value = 0.0 ;
    input.neurons[1]->value = 0.0 ;
    input.feedforward(); 

    layer output ( 1, layer::output ) ;
    output.neurons[0]->value = 0.0 ;
    output.feedforward() ;
    
    double cost = my_net.evaluate( &input , &output ) ; 

    layer * target = &output ;

    my_net.calculate_gradients( &input, &target ) ;

    std::cout << "cost : " << cost << " \n" ;

    
    
    // wait();
    
    return 0;
}

