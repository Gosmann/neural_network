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

    my_net.layers[0]->neurons[0]->value = 0 ;
    my_net.layers[0]->neurons[1]->value = 0 ;
    //my_net.layers[0]->feedforward() ; 

    // calculates values all the way from start to finish
    my_net.feedforward() ;  
    
    // displays interesting info
    my_net.summary() ;
        
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
    
    layer output ( 1, layer::output ) ;
    output.neurons[0]->activated = 0.0 ;
    
    double cost = my_net.evaluate( &input , &output ) ; 
    std::cout << "cost : " << cost << " \n" ;

    layer * target = &output ;

    my_net.calculate_gradients( &input , target ) ;


    /*
    std::cout << "layer 2 neuron 0 : \n";
    std::cout << "gradient : " << my_net.layers.back()->neurons[0]->gradient << "\n";
    
    std::cout << "layer 1 neuron 0 : \n";
    std::cout << "gradient : " << my_net.layers[1]->neurons[0]->gradient << "\n";
    
    std::cout << "layer 1 neuron 1 : \n";
    std::cout << "gradient : " << my_net.layers[1]->neurons[1]->gradient << "\n";
    */
   
    double learning_rate = 0.01 ;
    my_net.calculate_delta_weights( learning_rate ) ;  

    /*
    std::cout << "layer 2 neuron 0 (delta_weights): \n";
    std::cout << "delta 0 : " << (my_net.layers.back()->neurons[0]->delta_weights[0])[0][0] << "\n";
    std::cout << "delta 1 : " << (my_net.layers.back()->neurons[0]->delta_weights[1])[0][0] << "\n";
    std::cout << "delta 2 : " << (my_net.layers.back()->neurons[0]->delta_weights[2])[0][0] << "\n";
    */

    //updating

    // wait();
    my_net.layers.back()->neurons[0]->weights[0] += (my_net.layers.back()->neurons[0]->delta_weights[0])[0][0] ;
    my_net.layers.back()->neurons[0]->weights[1] += (my_net.layers.back()->neurons[0]->delta_weights[1])[0][0] ;
    my_net.layers.back()->neurons[0]->weights[2] += (my_net.layers.back()->neurons[0]->delta_weights[2])[0][0] ;

    cost = my_net.evaluate( &input , &output ) ; 
    std::cout << "cost : " << cost << " \n" ;


    return 0;
}

