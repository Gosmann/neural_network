#include <iostream>
#include <random>


#include "neural.hpp"

int main(){

    neural_net my_net ;     // defualt initialization of the class

    // adds a layer with 784 ( 28 * 28 ) neurons
    my_net.add_input_layer( 2 ) ;   

    my_net.add_hidden_layer( 4 ) ;

    my_net.add_output_layer( 1 ) ;    // adds output layer

    // creates the networks weights
    my_net.compile() ;

    my_net.layers[0]->neurons[0]->value = 0 ;
    my_net.layers[0]->neurons[1]->value = 0 ;
    
    // calculates values all the way from start to finish
    my_net.feedforward() ;  
    
    // displays interesting info
    //my_net.summary() ;
        
    dataset xor_dataset ;  

    int i;
    
    // training loop
    for(i = 0 ; i < 10000 ; i++){
        
        int index = ( i % 4 ) ; 

        my_net.gradient_descent( xor_dataset.data, xor_dataset.labels, 32, 1 );

        double cost = my_net.evaluate( xor_dataset.data[index], xor_dataset.labels[index] ) ;
        
        if( (i % 100) == 0) 
            std::cout << i << " , " << cost << " \n" ;

    }

    // testing myself
    layer * test_input = new layer( 2, layer::input ) ;

    // apply inputs to netork
    
    my_net.layers[0]->neurons[0]->value = 1.0 ;
    my_net.layers[0]->neurons[1]->value = 1.0 ;
    my_net.layers[0]->neurons[0]->activated = 1.0 ;
    my_net.layers[0]->neurons[1]->activated = 1.0 ;
    my_net.feedforward() ; 

    double cost = my_net.evaluate( my_net.layers[0] ,  xor_dataset.labels[ 0 ]) ; 

    //std::cout << cost << " \n"; 
    std::cout << my_net.layers.back()->neurons[0]->activated << " \n" ;
    //std::cout << my_net.layers.back()->neurons[0]->value << " \n" ;

    my_net.layers[0]->neurons[0]->value = 0.0 ;
    my_net.layers[0]->neurons[1]->value = 1.0 ;
    my_net.layers[0]->neurons[0]->activated = 0.0 ;
    my_net.layers[0]->neurons[1]->activated = 1.0 ;
    
    my_net.feedforward() ; 

    cost = my_net.evaluate( my_net.layers[0] ,  xor_dataset.labels[ 3 ]) ; 

    //std::cout << cost << " \n"; 
    std::cout << my_net.layers.back()->neurons[0]->activated << " \n" ;
    //std::cout << my_net.layers.back()->neurons[0]->value << " \n" ;
    

    return 0;
}

