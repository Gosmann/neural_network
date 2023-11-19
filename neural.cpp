#include <iostream>
#include <random>

#include "neural.hpp"

void print_hello(){

    std::cout << "Hello from lib! \n";
    
}

// constructors

// default neural_net constructor
neural_net::neural_net() {
    
    // any neural networks starts out with zero layers         

} ;

void neural_net::add_input_layer( int num_of_neurons ){

    // heap allocation
    layer * new_layer = new layer ( num_of_neurons ) ;
    new_layer->layer_type = layer::input ;
    new_layer->index = layers.size() ;
    new_layer->my_net = this ;
    
    // stack allocation
    //layer new_layer (num_of_neurons, &( layers[ 0 ] ) ) ;
    //new_layer.layer_type = layer::input ;
    //new_layer.index = layers.size();
    
    // adds 1 layer with a specific num of neurons
    layers.push_back( new_layer );
      
} 

void neural_net::add_hidden_layer( int num_of_neurons ){

    // heap allocation
    layer * new_layer = new layer ( num_of_neurons ) ;
    new_layer->layer_type = layer::hidden ;
    new_layer->index = layers.size() ;
    new_layer->my_net = this ;
    
    // stack allocation
    //layer new_layer (num_of_neurons, &( layers[ 0 ] ) ) ;
    //new_layer.layer_type = layer::input ;
    //new_layer.index = layers.size();
    
    // adds 1 layer with a specific num of neurons
    layers.push_back( new_layer );
    

} 

void neural_net::add_output_layer( int num_of_neurons ){

    // heap allocation
    layer * new_layer = new layer ( num_of_neurons ) ;
    new_layer->layer_type = layer::output ;
    new_layer->index = layers.size() ;
    new_layer->my_net = this ;
    
    // stack allocation
    //layer new_layer (num_of_neurons, &( layers[ 0 ] ) ) ;
    //new_layer.layer_type = layer::input ;
    //new_layer.index = layers.size();
    
    // adds 1 layer with a specific num of neurons
    layers.push_back( new_layer );
    
    

} 

void neural_net::compile(){
    
    int i;
    
    for( i = layers.size() - 1 ; i > 0 ; i-- ){
        
        //std::cout << layers[i]->neurons.size() << " \n" ;
        
        layers[i]->compile(  );    // compile meaning "define weigths"
    }

}

void neural_net::summary(){
    
    int i, j, k;

    //std::cout << "start compiling neural_net : \n" ;
    int total_weights = 0 ;

    std::cout << "artificial neural network parameters: \n" ;
    std::cout << "\t layers: " << layers.size() << "\n";
    for(i = 0 ; i < layers.size() ; i++){
        std::cout << "\t\t layer : [ " << i << " ] : [ " << layers[i]->neurons.size() << " ] \n" ;
        
        for( j = 0 ; j < layers[i]->neurons.size() ; j++ ){
            std::cout << "\t\t\t neuron : [ " << j << " ] : [ " << layers[i]->neurons[j]->weights.size() << " ] \n" ;

            for( k = 0 ; k < layers[i]->neurons[j]->weights.size() ; k++ ){
                std::cout << "\t\t\t\t weigth : [ " << k << " ] : [ " << layers[i]->neurons[j]->weights[k] << " ] \n" ;
                ++total_weights;
            }
        }
    }

    std::cout << "Total parameters to learn : " << total_weights << "\n" ;

}

void neural_net::feedforward() {

    int i ;

    for( i = 1 ; i < layers.size() ; i++ ){
        
        std::cout << "\n" << "feedforward layer : " << i << " \n";

        layers[i]->feedforward();

    }

}


// default layer constructor
layer::layer( int num_of_neurons ) {
        
    // any layer starts out with type input    
    layer::layer_type = layer::input ;
    
    // creates a layer with specified number of neurons
    //std::vector<neuron> neurons_vect (num_of_neurons) ;
    int i;

    for(i = 0 ; i < num_of_neurons ; i++ ){
        
        // heap allocation
        neuron * new_neuron = new neuron( this ) ;  
        new_neuron->index = neurons.size() ;

        // stack allocation
        //neuron new_neuron( creator ) ;
        //new_neuron.index = neurons.size() ;

        neurons.push_back( new_neuron ) ;         

    }

} ;

void layer::compile(){
    int i;

    //std::cout << "start compiling layer num " << index << " \n" ;

    for( i = 0 ; i < neurons.size() ; i++){

        //std::cout << "compiling : " << neurons[i].weights.size() << "\n" ;
    
        neurons[i]->compile(  );
        
    }
}

void layer::feedforward(){
    int i;

    // feedforwards for all neurons
    for( i = 0 ; i < neurons.size() ; i++){

        std::cout << "feedforwarding neuron: " << i << "\n" ;
    
        //layer * prev_layer = my_net->layers[ index - 1 ] ;

        neurons[i]->feedforward(  );
        
    }

}

// default neuron constructor
neuron::neuron( layer * creator ){
    
    // defines standard activation function
    neuron::neuron_activation = neuron::sigmoid ;

    my_layer = creator ;
    
    value = 0 ;

    // holds weight data
    //weights = std::vector<double> ( 0 ) ;
    

    
} ;

void neuron::compile(  ){

    int i;
    
    
    //std::cout << "start compiling neuron num " << index << " \n" ;

    int prev_layer_index = my_layer->index - 1 ;

    layer * prev_layer = my_layer->my_net->layers[ prev_layer_index ] ;
    
    // testing random number generator
    //std::default_random_engine generator(1002);
    // std::cout <<  (uint64_t)this << " \n";

    // random number based on memory access
    std::default_random_engine generator( (uint64_t)this );     
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    for(i = 0 ; i < prev_layer->neurons.size() + 1; i++ ){
        
        double number = distribution(generator) ;
        
        weights.push_back( number ) ;
        //std::cout << number << " \n";

    }
    
    // add bias term (adds above)
    // weights.push_back( 1.0 ) ;
    //std::cout << 1.0 << " \n";   


}


void neuron::feedforward() {

    int i ;

    value = 0;

    for( i = 0 ; i < weights.size() - 1; i++ ){
        
        //std::cout << "actuall neuron " << i << "\n";

        layer * prev_layer = my_layer->my_net->layers[ my_layer->index - 1 ] ;

        value += weights[i] * prev_layer->neurons[i]->value ;

    }

    value += weights[i] * 1.0 ;     // bias term
    
    //std::cout << value << "\n" ;

}
