# <p style="color:red">Neural Networks : The basics made for all</p>
By Bastien Lhopitallier and Robin Sobczyk

This course aims to make the following knowledge accesible to all :
- what is an artificial neuron and how it is linked to biological neurons
- what is a neural network
- the main different types of neural networks
- how training is done for neural networks, and what are the different options for training
- how to code a neural network from scratch
- how to code your first neural network with PyTorch and train it
- how to create and read a computation flow of a neural network
- how to debug your neural network through gradient inspection
- what are the risks and limits of neural networks
- how to run basic analysis of the activations with heatmaps
- the ethical concerns raised by deep learning

À faire :
- [x] faire un vrai sommaire avec un vrai découpage
- [x] détailler chaque point du sommaire
- [x] mettre une licence
- [ ] regarder comment share (gitfront.io ?)
- [x] ajouter un `requirements.txt`
- [ ] ajouter tous les modules dans le `requirements.txt`
- [ ] ajouter les versions des modules dans le `requirements.txt`
- [x] mettre le code sous forme de module
- [x] nettoyer le code dans le module
- [x] aligner le readme avec les summary des cours
- [x] préciser les versions de python recommandées dans le cours 0

### Table of content  
Course 0: Introduction
- what is an artificial neuron and how it is linked to biological neurons
- how to assemble artificial neurons, what is an artificial neural network
- the different types of networks and their use case in a quick overview

Course 1: Your first networks
- how do we train a neural network (forward pass, backpropagation)
- how to code a neural network from scratch
- PyTorch and autograd (automatic differentiation)
- how to setup the training on a CPU or a GPU
- how to code a neural network with PyTorch and train it
- how to save and load your model
- classic forms of training

Course 2: How to properly read and present your networks and their results
- how to make your results reproducible
  - how to seed your code
  - how to enable reproducibility on GPU
  - what are the uncontrollable factors that might undermine your reproducibility
  - how much providing source code is necessary
- how to analyse the results of your network
  - what is overfitting and underfitting
  - how to identify them
- how to represent your network to present it to other people
  - how to read and create the computation flow of a neural network
  - how to represent neural networks from that control flow
- how to avoid common problems
  - what is gradient vanishing and exploding
  - how to use gradient inspection to locate them

Course 3: Technical limits of AI
- explainability of a neural network model
- trust in its results and capacity to correctly solve its task (proxy values)
- how to realise a basic analysis of the activations of a network with heatmaps
- an example of one of the risks of neural networks: malicious sample optimization

Course 4: Ethical considerations
- ethical considerations about
  - datasets
  - privacy in deep learning
  - generative AI
  - AGI

Course 5: Going further
- getting a vague idea of how vast is the world of deep learning

### License

This course is distributed under the [Academic Public License](LICENSE.txt)