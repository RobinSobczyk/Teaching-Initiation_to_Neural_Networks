# <p style="color:red">Neural Networks : The basics made for all</p>
By Bastien Lhopitallier and Robin Sobczyk

In this course, you will learn :
- what is an artificial neuron and how it is linked to biological neurons
- what is a neural network
- how training is done for neural network, and what are the different options for training
- the main different types of neural networks
- how to code a neural network from scratch
- how to code your first neural network with PyTorch and train it
- how to create and read a computation flow of a neural network
- how to debug your neural network through gradient inspection
- what are the risks of neural networks with malicious sample optimisation
- how to run basic analysis of the activations with heatmaps
- what are the limits of neural networks with some measures of random effects

À faire :
- [x] faire un vrai sommaire avec un vrai découpage
- [x] détailler chaque point du sommaire
- [ ] mettre une licence
- [ ] regarder comment share (gitfront.io ?)
- [x] ajouter un `requirements.txt`
- [ ] ajouter tous les modules dans le `requirements.txt`
- [ ] ajouter les versions des modules dans le `requirements.txt`
- [x] mettre le code sous forme de module
- [ ] nettoyer le code dans le module
- [ ] aligner le readme avec les summary des cours
- [ ] préciser les versions de python recommandées dans le cours 0

### Sommaire provisoire :  
Cours 0 : Introduction :
- what is an artificial neuron and how it is linked to biological neurons
- how to assemble artificial neurons, what is a network, still some links with the brain
- different types of networks and use (quick overview of different type of tasks, type of operations, use in other frameworks like RL)

Cours 1 : Les premiers réseaux :
- how do we train a neural network (forward pass, backpropagation)
- how to code a neural network from scratch
- PyTorch and autograd (automatic differentiation)
- how to setup the training on a CPU or a GPU
- how to code a neural network with PyTorch and train it
- how to save and load your model
- classic forms of training

Cours 2 : présenter, représenter et lire les réseaux, leurs résultats
- Overfitting, underfitting
- savoir identifier l'overfitting
- how to create and read a computation flow of a neural network
- how to represent neural networks along their computation flow (représentations diverses, CNN etc)
- how to debug your neural network through gradient inspection, gradient vanishing and exploding
- seeding, source code and reproducibility
- GPU and reproducibility

Cours 3 : Les limites techniques de l'IA :
- explainability of a neural network model
- trust in its results and capacity to correctly solve its task (proxy values)
- how to realise a basic analysis of the activations of a network with heatmaps
- an example of one of the risks of neural networks: malicious sample optimization

Cours 4 : Ethique sociale :
- les datasets et l'IA générative
- les AGI
- privacy

Cours 5 : Approfondissement :
- types de réseaux et use case :
  - classification et linear networks
  - images et CNN (audio as images)
  - encoder/decoder, modélisation de proba
  - NLP, traitement du langage, tokens, embeddings, attention, LLMs ?
  - RL et apprentissage
  - séquences et RNN
  - GNN
  - SNN
  - Geometry and Topology ?
  - Federated learning ?
- tricks
  - residual connections
  - dense blocks
  - types of loss (contrastive etc)
  - diffusion models
  - dataset enlarging (rotate, crop etc)
  - génération de résultat par sampling d'une loi de proba
  - mémoire ? (vector database, in context learning, Hopfield)
- theory ?
  - JAX/torch.func, categories ?
  - re auto-diff
  - densité
- ressources utiles