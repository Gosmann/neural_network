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
    
    // creates weights
    my_net.compile() ;

    my_net.layers[0]->neurons[0]->value = 0 ;
    my_net.layers[0]->neurons[1]->value = 0 ;
    //my_net.layers[0]->feedforward() ; 

    // calculates values all the way from start to finish
    my_net.feedforward() ;  
    
    // displays interesting info
    //my_net.summary() ;
        
    dataset xor_dataset ;  

    int i;
    for(i = 0 ; i < 10000 ; i++){
        
        //my_net.summary() ;

        //std::srand( i ) ;
        //int random_index = ( std::rand() % 4 ) ;
        int random_index = i % 4 ; 

        layer * input = xor_dataset.data[ random_index ] ;         // generates nums from 0 to 3
        layer * output = xor_dataset.labels[ random_index ] ;      // generates nums from 0 to 3
        
        //std::cout << i << "\n"; 

        double cost = my_net.evaluate( input , output ) ; 
        
        /*
        std::cout << "layer 0 neuron 0 : activated : " << my_net.layers[0]->neurons[0]->activated << "\n";
        std::cout << "layer 0 neuron 1 : activated : " << my_net.layers[0]->neurons[1]->activated << "\n" ;
        
        
        std::cout << "layer 1 neuron 0 : output : " << my_net.layers.back()->neurons[0]->value << "\n";
        std::cout << "layer 1 neuron 0 : activated : " << my_net.layers.back()->neurons[0]->activated << "\n" ;
        std::cout << "layer 1 neuron 0 : cost : " << cost << "\n";
        */

        if( i % (100+1) == 0 ){
            std::cout << i << " , " << cost << "\n";
        }

        layer * target = output ;

        my_net.calculate_gradients( input , target ) ;

        //std::cout << "layer 1 neuron 0 : gradient : " << my_net.layers.back()->neurons[0]->gradient << "\n";

        double learning_rate = 1 ;
        my_net.calculate_delta_weights( learning_rate ) ;  

        /*
        std::cout << "layer 1 neuron 0 weight 0 : delta : " << 
            my_net.layers.back()->neurons[0]->delta_weights[0][0][0] << "\n" ;
        std::cout << "layer 1 neuron 0 weight 1 : delta : " << 
            my_net.layers.back()->neurons[0]->delta_weights[1][0][0] << "\n";
        std::cout << "layer 1 neuron 0 weight 2 : delta : " << 
            my_net.layers.back()->neurons[0]->delta_weights[2][0][0] << "\n";
        */

        my_net.apply_delta_weights();
        
    }

    // testing myself
    layer * test_input = new layer( 2, layer::input ) ;

    // apply inputs to netork
    
    my_net.layers[0]->neurons[0]->value = 0.0 ;
    my_net.layers[0]->neurons[1]->value = 0.0 ;
    my_net.layers[0]->neurons[0]->activated = 0.0 ;
    my_net.layers[0]->neurons[1]->activated = 0.0 ;
    my_net.feedforward() ; 

    double cost = my_net.evaluate( my_net.layers[0] ,  xor_dataset.labels[ 0 ]) ; 

    //std::cout << cost << " \n"; 
    std::cout << my_net.layers.back()->neurons[0]->activated << " \n" ;
    //std::cout << my_net.layers.back()->neurons[0]->value << " \n" ;

    my_net.layers[0]->neurons[0]->value = 1.0 ;
    my_net.layers[0]->neurons[1]->value = 0.0 ;
    my_net.layers[0]->neurons[0]->activated = 1.0 ;
    my_net.layers[0]->neurons[1]->activated = 0.0 ;
    
    my_net.feedforward() ; 

    cost = my_net.evaluate( my_net.layers[0] ,  xor_dataset.labels[ 3 ]) ; 

    //std::cout << cost << " \n"; 
    std::cout << my_net.layers.back()->neurons[0]->activated << " \n" ;
    //std::cout << my_net.layers.back()->neurons[0]->value << " \n" ;
    

    return 0;
}

