#include <iostream>
#include <random>
#include <chrono>
#include <thread>

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
    layer * new_layer = new layer ( num_of_neurons, layer::hidden ) ;
    
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
        
        //std::cout << "\n" << "feedforward layer : " << i << " \n";

        layers[i]->feedforward();

    }

}

double neural_net::evaluate( layer * input, layer * output ){
    
    int i;
    double loss = 0;
    
    // apply inputs to netork
    for( i = 0 ; i < input->neurons.size() ; i++ ){
        layers[0]->neurons[i]->value = input->neurons[i]->value ; 
        //layers[0]->neurons[i]->activated = input->neurons[i]->activated ; 
    }
    
    //std::cout << "layer 0 size (must be 2) : " << layers[0]->neurons.size() << " \n" ;

    // feedforwads all input changes
    feedforward() ;

    //std::cout << "* layer 0 neuron 0 : activated : " << layers[0]->neurons[0]->activated << "\n";
    //std::cout << "* layer 0 neuron 1 : activated : " << layers[0]->neurons[1]->activated << "\n" ;  

    //output->feedforward();

    for(i = 0 ; i < output->neurons.size() ; i++ ){

        //std::cout << "target : " << output->neurons[i]->activated << " \n";
        //std::cout << "actual : " << layers.back()->neurons[0]->activated << " \n";

        double error = output->neurons[i]->activated - layers.back()->neurons[i]->activated ; 

        //std::cout << "Error : " << error << " \n";

        loss += (error * error) ;

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

    //target->feedforward();

    // calculate gradient for each layer (back to front)
    for( i = layers.size() - 1 ; i > 0 ; i-- ){
    
        layers[i]->calculate_gradients( target ) ; 

    }

}

void neural_net::calculate_delta_weights( double learning_rate ){

    int i;
    
    // calculate delta weights for each layer (back to front)
    for( i = layers.size() - 1 ; i > 0 ; i-- ){
    
        layers[i]->calculate_delta_weights( learning_rate ) ; 

    }

}

void neural_net::apply_inputs( layer * input_layer ){

    int i; 

    for( i = 0 ; i < input_layer->neurons.size() ; i++ ){

        layers[0]->neurons[i]->value = input_layer->neurons[i]->value ; 

    }

    feedforward();

}

void neural_net::apply_delta_weights( void ){

    int i;
    
    // calculate delta weights for each layer (back to front)
    for( i = layers.size() - 1 ; i > 0 ; i-- ){
    
        layers[i]->apply_delta_weights(); 

    }

}

// default layer constructor
layer::layer( int num_of_neurons ) {
        
    // any layer starts out with type input    
    //layer::layer_type = layer::input ;
    
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
                new_neuron->neuron_activation = neuron::sigmoid ;    
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

    //std::cout << "size(must be 2): " << neurons.size() << "\n" ;

    // feedforwards for all neurons
    for( i = 0 ; i < neurons.size() ; i++){

        //std::cout << "feedforwarding neuron: " << i << " - type:  " << 
        //    layer_type <<" \n" ;
        
        //layer * prev_layer = my_net->layers[ index - 1 ] ;
        if( layer_type == layer::input ){
            neurons[i]->input_feedforward(  );
        }
        else{
            neurons[i]->feedforward(  );
        }
        
        
    }

}

void layer::calculate_gradients( layer * target ){
    
    int i;

    // considers all neurons in the output layer    
    for( i = 0 ; i < neurons.size() ; i++ ){
        neurons[i]->calculate_gradient( target->neurons[i] ) ;
    }

} 

void layer::calculate_delta_weights( double learning_rate ){

    int i;

    // calculate delta weights for all neurons
    for( i = 0 ; i < neurons.size() ; i++){

        neurons[i]->calculate_delta_weights( learning_rate );
        
    }

}

void layer::apply_delta_weights( void ){

    int i;

    // calculate delta weights for all neurons
    for( i = 0 ; i < neurons.size() ; i++){

        neurons[i]->apply_delta_weights(  );
        
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
    //std::default_random_engine generator( 1234 );     
    std::uniform_real_distribution<double> distribution(-0.5,0.5);

    for(i = 0 ; i < prev_layer->neurons.size() + 1; i++ ){
        
        double number = distribution(generator) ;
        
        weights.push_back( number ) ;
        //std::cout << number << " \n";

    }

    // create delta_weights_vectors
    for(i = 0 ; i < weights.size() ; i++ ){

        std::vector<double> * deltas = new std::vector<double> ;

        delta_weights.push_back( deltas ) ; 

    }
    
    // add bias term (adds above)
    // weights.push_back( 1.0 ) ;
    //std::cout << 1.0 << " \n";   
}

void neuron::feedforward(  ) {

    int i ;

    value = 0 ;

    // TODO make this better
    for( i = 0 ; i < weights.size() ; i++ ){
    
        //std::cout << "actuall neuron " << i << "\n";

        // last iteration
        if( i + 1 == weights.size() ){
            value += weights[i] * 1.0 ;     // bias term
        }

        else{
            layer * prev_layer = my_layer->my_net->layers[ my_layer->index - 1 ] ;
            value += weights[i] * prev_layer->neurons[i]->activated ;
        }

    }


    // calculate activation
    switch( neuron_activation ){
        case sigmoid :
            activated = 1.0 / ( 1.0 + std::exp( -1.0 * value ) ) ;
            break;
        
        case linear :
            // std::cout << "neuron::feedforward input \n";
            activated = value ; 
            break;

    }

    //std::cout << "feed forward neuron : " << value << "\n" ;

}

void neuron::input_feedforward(  ) {

    int i ;

    // calculate activation
    switch( neuron_activation ){
        case sigmoid :
            activated = 1.0 / ( 1.0 + std::exp( -1.0 * value ) ) ;
            break;
        
        case linear :
            // std::cout << "neuron::feedforward input \n";
            activated = value ; 
            break;

    }

    //std::cout << "feed forward neuron : " << value << "\n" ;

}

void neuron::calculate_gradient( neuron * target ){
    
    int i ;
    
    switch( my_layer->layer_type ){

        case layer::output :
            
            switch ( neuron_activation ) {
                case sigmoid:                    
                    gradient = ( activated - target->activated ) * 
                        activated * (1 - activated ) ;    
                    break;
                
                case linear:
                    gradient = ( activated - target->activated ) ;
                    break;
            }

            break ;
        
        case layer::hidden :
            
            double sum = 0 ;
            // considers all further neurons
            layer * next_layer = my_layer->my_net->layers[ my_layer->index + 1 ] ;

            for( int i = 0 ; i < next_layer->neurons.size() ; i++ ){

                sum += next_layer->neurons[i]->gradient * next_layer->neurons[i]->weights[ index ] ;

            }

            switch ( neuron_activation ) {
                case sigmoid:                    
                    gradient = sum * activated * (1 - activated ) ;    
                    break;
                
                case linear:
                    gradient = sum ;
                    break;
            }

            break ;

    }

}

void neuron::calculate_delta_weights( double learning_rate ){
    
    int i;

    double prev_activated = 0;

    // calculates for all weights
    for( i = 0 ; i < weights.size() ; i++ ){
    
        // last iteration
        if( ( i + 1 ) == weights.size() ){
            // bias
            prev_activated = 1.0 ;
        }

        else{
            layer * prev_layer = my_layer->my_net->layers[ my_layer->index - 1 ] ;
            
            prev_activated = prev_layer->neurons[ i ]->activated ; 
        
        }

        double delta_weight = -1.0 * learning_rate * gradient * prev_activated ;

        delta_weights[i]->push_back( delta_weight ) ;

    }
  
}

void neuron::apply_delta_weights(void){

    int i, j;

    double prev_activated = 0;

    // evaluates one weight at a time
    for( i = 0 ; i < delta_weights.size() ; i++ ){
    
        double sum = 0 ;
        int batch_size = delta_weights[i]->size();

        // evalates all deltas from a batch
        /*
        for( j = ( batch_size ); j > 0 ; j-- ){
            sum += delta_weights[i][0][j - 1] ;  
            delta_weights[i][0].pop_back() ;  
        }
        
        double average = ( sum / (double)(batch_size) ) ;
        */

        weights[i] += delta_weights[i][0][0]  ;
        delta_weights[i][0].pop_back() ;  

    }

}

dataset::dataset( void ){

    create_xor() ; 

}

void dataset::create_xor( void ){

    layer * input = new layer( 2, layer::input ) ;
    input->neurons[0]->value = 0.0 ;
    input->neurons[1]->value = 0.0 ;
    data.push_back( input );

    input = new layer( 2, layer::input ) ;
    input->neurons[0]->value = 0.0 ;
    input->neurons[1]->value = 1.0 ;
    data.push_back( input );

    input = new layer( 2, layer::input ) ;
    input->neurons[0]->value = 1.0 ;
    input->neurons[1]->value = 0.0 ;
    data.push_back( input );

    input = new layer( 2, layer::input ) ;
    input->neurons[0]->value = 1.0 ;
    input->neurons[1]->value = 1.0 ;
    data.push_back( input );

    //
    layer * output = new layer( 1, layer::output ) ;
    output->neurons[0]->activated = 0.0 ;
    labels.push_back( output );

    output = new layer( 1, layer::output ) ;
    output->neurons[0]->activated = 1.0 ;
    labels.push_back( output );

    output = new layer( 1, layer::output ) ;
    output->neurons[0]->activated = 1.0 ;
    labels.push_back( output );

    output = new layer( 1, layer::output ) ;
    output->neurons[0]->activated = 0.0 ;
    labels.push_back( output );


}