#include <iostream>
#include <random>

#include "neural.hpp"

void print_hello(){

    std::cout << "Hello from lib! \n";
    
}

void wait(){
    
    while(1){
        int i;
        for (i = 0 ; i < 3 ; i++){
            std::cout << "." ;
            std::cout.flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        std::cout << "\r" ;
        
        for (i = 0 ; i < 10 ; i++){
            std::cout << " " ;
            std::cout.flush();
        }
        std::cout << "\r" ;
        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

    }

}


// constructors
// default neural_net constructor
neural_net::neural_net() {
    
    // any neural networks starts out with zero layers         

} ;

void neural_net::add_input_layer( int num_of_neurons ){

    // heap allocation
    layer * new_layer = new layer ( num_of_neurons, layer::input ) ;
    
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
    layer * new_layer = new layer ( num_of_neurons, layer::output ) ;
    
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

            if( i == 0 ){   // print input values
                std::cout << "\t\t\t\t input  : [ " << j << " ] : [ " << layers[i]->neurons[j]->activated << " ] \n" ;    
            }

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

    for( i = 0 ; i < layers.size() ; i++ ){
        
        std::cout << "\n" << "feedforward layer : " << i << " \n";

        layers[i]->feedforward();

    }

}

double neural_net::evaluate( layer * input, layer * output ){
    
    int i;
    double loss = 0;
    
    // apply inputs to netork
    for( i = 0 ; i < input->neurons.size() ; i++ ){
        layers[0]->neurons[i]->value = input->neurons[i]->value ; 
        layers[0]->neurons[i]->activated = input->neurons[i]->activated ; 
    }
    

    for(i = 0 ; i < output->neurons.size() ; i++ ){

        double error = output->neurons[i]->activated - layers.back()->neurons[i]->activated ; 

        loss += error * error ;

    }

    loss *= 0.5 ;   // see formula for details

    return loss ;
}

void neural_net::calculate_gradients( layer * input, layer * target ){

    int i;
    
    // apply inputs to the network
    for( i = 0 ; i < input->neurons.size() ; i++ ){
        layers[0]->neurons[i]->value = input->neurons[i]->value ; 
        layers[0]->neurons[i]->activated = input->neurons[i]->activated ; 
    }

    // feedforwads all input changes
    feedforward(); 

    // calculate gradient for each layer (back to front)
    for( i = layers.size() - 1 ; i > 0 ; i-- ){
    
        layers[i]->calculate_gradients( layer * target ) ; 

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

// default layer constructor
layer::layer( int num_of_neurons, layer::type input_layer_type ) {
        
    // any layer starts out with type input    
    layer_type = input_layer_type ;
    
    // creates a layer with specified number of neurons
    //std::vector<neuron> neurons_vect (num_of_neurons) ;
    int i;

    for(i = 0 ; i < num_of_neurons ; i++ ){
        
        // heap allocation
        neuron * new_neuron = new neuron( this ) ;  
        new_neuron->index = neurons.size() ;

        switch( layer_type ){
            case input:
                new_neuron->neuron_activation = neuron::linear ;    
                break;
            case hidden:
                new_neuron->neuron_activation = neuron::sigmoid ;    
                break;
            case output:
                new_neuron->neuron_activation = neuron::linear ;    
                break;
        }

        // stack allocation
        //neuron new_neuron( creator ) ;
        //new_neuron.index = neurons.size() ;

        neurons.push_back( new_neuron ) ;         

    }

} ;

void layer::compile( ){
    int i;

    //std::cout << "start compiling layer num " << index << " \n" ;

    for( i = 0 ; i < neurons.size() ; i++){

        //std::cout << "compiling : " << neurons[i].weights.size() << "\n" ;
    
        neurons[i]->compile(  );
        
    }
}

void layer::feedforward( ){
    int i;

    // feedforwards for all neurons
    for( i = 0 ; i < neurons.size() ; i++){

        std::cout << "feedforwarding neuron: " << i << " - type:  " << 
            layer_type <<" \n" ;
        
        //layer * prev_layer = my_net->layers[ index - 1 ] ;

        neurons[i]->feedforward(  );
        
    }

}

void layer::calculate_gradients( layer * target ){
    
    int i;

    // considers all neurons in the output layer    
    for( i = 0 ; i < neurons.size() ; i++ ){
        neurons[i]->calculate_gradients_output( target->neurons[i] ) ;
    }

} 

// default neuron constructor
neuron::neuron( layer * creator ){
    
    // defines standard activation function
    
    my_layer = creator ;
    
    value = 0 ;
    activated = 0 ;
    gradient = 0 ; 

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


    if( weights.size() > 0 ){

        for( i = 0 ; i < weights.size() - 1; i++ ){
        
            //std::cout << "actuall neuron " << i << "\n";

            layer * prev_layer = my_layer->my_net->layers[ my_layer->index - 1 ] ;

            value += weights[i] * prev_layer->neurons[i]->activated ;

        }

        value += weights[i] * 1.0 ;     // bias term
    
    }


    // calculate activation
    switch( neuron_activation ){
        case sigmoid :
            activated = 1.0 / ( 1.0 + std::exp( -1.0 * value ) ) ;
            break;
        
        case linear :
            std::cout << "neuron::feedforward output \n";
            activated = value ; 
            break;

    }

    //std::cout << value << "\n" ;

}

void neuron::calculate_gradient( neuron * target ){
    
    int i ;

    switch( my_layer->layer_type ){

        case layer::output :
            
            gradient = (activated - target->activated) * activated * (1 - activated ) ;

            break ;
        
        case layer::input :
            
            double sum = 0 ;
            // considers all further neurons

            break ;

    }

}