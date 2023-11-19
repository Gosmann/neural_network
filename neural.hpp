#include <vector>
#include <random>

// functions
void print_hello();

/*  things the library should implement

    neural_net
    layer
    neuron
    weight


*/ 

// classes
class neuron ;
class layer ;
class neural_net ;

class neuron {

    public :
        // <random>
        //std::default_random_engine generator;
        //std::uniform_real_distribution<double> distribution;

        // TODO declare this enum outiside class declaration
        enum activation {       
            sigmoid             
        };

        // choose neuron activation function
        activation neuron_activation ;
        
        layer * my_layer ;
        int index ;

        double value ; 

        // vector for holding the weigths with respect to this neuron       
        std::vector<double> weights ;

        // constructor
        neuron( layer * ) ;

        // compile neuron (attributes weights to it)
        void compile(  ) ;

        void feedforward(  ) ;

} ;


class layer {

    public :

        // TODO declare this enum outside class declaration        
        enum type {             
            input, hidden, output 
        } ;

        // holds the type of the layer
        type layer_type ; 
        int index ;
        const neural_net * my_net ;

        // vector for holding the layer's neurons's data
        std::vector<neuron *> neurons ; 

        // constructor
        layer( int ) ;      // a layer always has a number of neurons

        // compile a layer (assigns weights)
        void compile();

        // calculates neuron values all the way from start to finish
        void feedforward( void ) ;
} ;


class neural_net {
    
    public :
        
        // holds the list for the neural net's layers references
        std::vector<layer *> layers ;

        // constructors 
        neural_net() ;              // no argument

        // adds an input layer to the network
        void add_input_layer( int ) ;

        // adds a hidden layer to the network
        void add_hidden_layer( int ) ;

        // adds an output layer to the network
        void add_output_layer( int ) ;

        // creates the netork's weights
        void compile( void );

        // displays general info
        void summary( void );

        // calculates neuron values all the way from start to finish
        void feedforward( void );
} ;

/*
neural_net::neural_net() ; 
void neural_net::add_input_layer( int num_of_neurons );
void neural_net::add_hidden_layer( int num_of_neurons );
void neural_net::add_output_layer( int num_of_neurons );

void neural_net::compile();
layer::layer( int num_of_neurons, neural_net * creator );
void layer::compile();

neuron::neuron( layer * creator2 );
void neuron::compile(  );
*/
