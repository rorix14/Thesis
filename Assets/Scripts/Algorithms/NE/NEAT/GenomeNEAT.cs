using System;
using System.Collections.Generic;
using NN.CPU_Single;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Algorithms.NE.NEAT
{
    public enum NeuronType
    {
        Input,
        Hidden,
        Output,
        None
    }

    public class GenomeNEAT
    {
        private int _genomeId;
        private List<LinkGene> _links;
        private List<NeuronGene> _neurons;
        private readonly int _inputNumber;
        private readonly int _outputNumber;

        public float Fitness;
        private float _adjustedFitness;
        private float _amountToSpawn;

        private List<float>[] _neuronsWeights;
        private List<int>[] _neuronsOutputIndexes;
        private float[] _neuronOutputs;

        public int GenomeSize => _links.Count;

        public float AdjustedFitness
        {
            get => _adjustedFitness;
            set => _adjustedFitness = value;
        }

        public LinkGene GetLinkGene(int index) => _links[index];
        public NeuronGene GetNeuronById(int id) => _neurons[GetElementPos(id)];

        // This might change
        private NEATNetworkModel _neatNetworkModel;

        public GenomeNEAT(int genomeId, int inputNumber, int outputNumber)
        {
            _genomeId = genomeId;
            _inputNumber = inputNumber;
            _outputNumber = outputNumber;

            _neurons = new List<NeuronGene>(inputNumber + outputNumber);
            var neuronId = 1;
            for (int i = 0; i < inputNumber; i++)
            {
                _neurons.Add(new NeuronGene(neuronId++, NeuronType.Input, 0));
            }

            for (int i = 0; i < outputNumber; i++)
            {
                _neurons.Add(new NeuronGene(neuronId++, NeuronType.Output, 1));
            }

            _links = new List<LinkGene>(inputNumber * outputNumber);
            var innovationNumber = 1;
            for (int i = 0; i < inputNumber; i++)
            {
                var inputNeuronId = _neurons[i].Id;
                for (int j = 0; j < outputNumber; j++)
                {
                    _links.Add(new LinkGene(innovationNumber++, inputNeuronId, _neurons[j].Id,
                        0.005f * Random.Range(-4f, 4f), true, false));
                }
            }
        }

        public GenomeNEAT(int genomeId, List<LinkGene> links, List<NeuronGene> neurons, int inputNumber,
            int outputNumber)
        {
            _genomeId = genomeId;
            _links = links;
            _neurons = neurons;
            _inputNumber = inputNumber;
            _outputNumber = outputNumber;
        }

        public GenomeNEAT(int genomeId, GenomeNEAT genomeToCopy)
        {
            _genomeId = genomeId;
            _links = new List<LinkGene>(genomeToCopy._links);
            _neurons = new List<NeuronGene>(genomeToCopy._neurons);
            _inputNumber = genomeToCopy._inputNumber;
            _outputNumber = genomeToCopy._outputNumber;
        }

        public void CreatePhenotype()
        {
            _neuronOutputs = new float[_neurons.Count];
            _neuronsWeights = new List<float>[_neurons.Count];
            _neuronsOutputIndexes = new List<int>[_neurons.Count];
            for (int i = 0; i < _neuronsWeights.Length; i++)
            {
                _neuronsWeights[i] = new List<float>();
                _neuronsOutputIndexes[i] = new List<int>();
            }

            for (int i = 0; i < _links.Count; i++)
            {
                var link = _links[i];
                if (!link.IsEnabled) continue;

                var toNeuron = link.ToNeuron;
                _neuronsWeights[GetElementPos(toNeuron)].Add(link.Weight);
                _neuronsOutputIndexes[GetElementPos(toNeuron)].Add(GetElementPos(link.FromNeuron));
            }
        }

        public float[] Forward(float[] input)
        {
            //TODO: reuse variables 
            var networkOutputs = new float[_outputNumber];
            var outputNeuronStart = _neurons.Count - _outputNumber - 1;
            for (int i = 0; i < _inputNumber; i++)
            {
                _neuronOutputs[i] = input[i];
            }

            for (int i = 0; i < _neuronsWeights.Length; i++)
            {
                var sum = 0f;
                for (int j = 0; j < _neuronsWeights[i].Count; j++)
                {
                    sum += _neuronsWeights[i][j] * _neuronOutputs[_neuronsOutputIndexes[i][j]];
                }

                var expPos = (float)Math.Exp(sum);
                var expNeg = (float)Math.Exp(-sum);
                var result = (expPos - expNeg) / (expPos + expNeg);
                _neuronOutputs[i] = result;

                if (i < outputNeuronStart) continue;

                networkOutputs[i % outputNeuronStart] = result;
            }

            return networkOutputs;
        }

        public void AddLink(float mutationRate, float recurrentChance, InnovationNEAT innovation, int numTrysToFindLoop,
            int numTrysToAddLink)
        {
            if (Random.value > mutationRate) return;

            var neuron1Id = -1;
            var neuron2Id = -1;
            var isRecurrent = false;

            if (Random.value < recurrentChance)
            {
                while (numTrysToFindLoop-- > 0)
                {
                    var neuronPos = Random.Range(_inputNumber, _neurons.Count);
                    var neuron = _neurons[neuronPos];
                    //TODO: might not need to check for neuron recurrence and neuron type 
                    if (DuplicateLink(neuron.Id, neuron.Id) || neuron.IsRecurrent ||
                        neuron.NeuronType == NeuronType.Input) continue;

                    neuron1Id = neuron.Id;
                    neuron2Id = neuron.Id;

                    neuron.IsRecurrent = true;
                    _neurons[neuronPos] = neuron;

                    isRecurrent = true;
                    numTrysToFindLoop = 0;
                }
            }
            else
            {
                while (numTrysToAddLink-- > 0)
                {
                    var neuron1 = _neurons[Random.Range(0, _neurons.Count)];
                    var neuron2 = _neurons[Random.Range(_inputNumber, _neurons.Count)];
                    neuron1Id = neuron1.Id;
                    neuron2Id = neuron2.Id;

                    //TODO: might also need to check for depth
                    if (!(DuplicateLink(neuron1Id, neuron2Id) || neuron1Id == neuron2Id))
                    {
                        //TODO: equals =, not in original implementation
                        if (neuron1.SplitX >= neuron2.SplitX)
                        {
                            isRecurrent = true;
                        }

                        numTrysToAddLink = 0;
                    }
                    else
                    {
                        neuron1Id = -1;
                        neuron2Id = -1;
                    }
                }
            }

            if (neuron1Id < 0 || neuron2Id < 0) return;

            var linkId = innovation.CheckInnovation(neuron1Id, neuron2Id, NeuronType.None);
            if (linkId < 0)
            {
                linkId = innovation.CreateNewInnovation(neuron1Id, neuron2Id, NeuronType.None);
                _links.Add(
                    new LinkGene(linkId, neuron1Id, neuron2Id, 0.005f * Random.Range(-4f, 4f), true, isRecurrent));
            }
            else
            {
                _links.Add(
                    new LinkGene(linkId, neuron1Id, neuron2Id, 0.005f * Random.Range(-4f, 4f), true, isRecurrent));
            }
        }

        public void AddNeuron(float mutationRate, InnovationNEAT innovation, int numTrysToFindOldLink)
        {
            if (Random.value > mutationRate) return;

            var done = false;
            var chosenLinkIndex = 0;
            var chosenLink = new LinkGene();

            while (!done)
            {
                chosenLinkIndex = Random.Range(0, _links.Count);
                chosenLink = _links[chosenLinkIndex];
                if (chosenLink.IsEnabled && !chosenLink.IsRecurrent)
                {
                    done = true;
                }
            }

            chosenLink.IsEnabled = false;
            _links[chosenLinkIndex] = chosenLink;

            var originalWeight = chosenLink.Weight;
            var fromNeuron = chosenLink.FromNeuron;
            var toNeuron = chosenLink.ToNeuron;
            var newDepth = (_neurons[GetElementPos(fromNeuron)].SplitX + _neurons[GetElementPos(toNeuron)].SplitX) / 2f;

            var newNeuronId = innovation.CheckInnovation(fromNeuron, toNeuron, NeuronType.Hidden);

            //TODO: if condition can be joined with the conditions bellow, but must switch conditions around   
            if (newNeuronId >= 0 && GetElementPos(newNeuronId) >= 0)
            {
                newNeuronId = -1;
            }

            if (newNeuronId < 0)
            {
                newNeuronId = innovation.CreateNewInnovation(fromNeuron, toNeuron, NeuronType.Hidden);
                _neurons.Add(new NeuronGene(newNeuronId, NeuronType.Hidden, newDepth));

                var link1Id = innovation.CreateNewInnovation(fromNeuron, newNeuronId, NeuronType.None);
                _links.Add(new LinkGene(link1Id, fromNeuron, newNeuronId, 1f, true, false));

                var link2Id = innovation.CreateNewInnovation(newNeuronId, toNeuron, NeuronType.None);
                _links.Add(new LinkGene(link2Id, newNeuronId, toNeuron, originalWeight, true, false));
            }
            else
            {
                var link1Id = innovation.CheckInnovation(fromNeuron, newNeuronId, NeuronType.None);
                var link2Id = innovation.CheckInnovation(newNeuronId, toNeuron, NeuronType.None);

                //TODO: temporary check, remove when it is clear this condition is never true
                if (link1Id < 0 && link2Id < 0)
                {
                    Debug.Log("While creating an existing neuron some links could not be found");
                    return;
                }

                _neurons.Add(new NeuronGene(newNeuronId, NeuronType.Hidden, newDepth));
                _links.Add(new LinkGene(link1Id, fromNeuron, newNeuronId, 1f, true, false));
                _links.Add(new LinkGene(link2Id, newNeuronId, toNeuron, originalWeight, true, false));
            }
        }

        public void MutateWeights(float mutationRate, float weightReplaceRate)
        {
            if (Random.value > mutationRate) return;
            
            for (int i = 0; i < _links.Count; i++)
            {
                //TODO: create buffers for both random variables, put those buffers in the NEAT Model class
                var link = _links[i];
                if (Random.value < weightReplaceRate)
                {
                    link.Weight = 0.005f * Random.Range(-4f, 4f);
                }
                else
                {
                    link.Weight += 0.01f * NnMath.RandomGaussian(-10, 10);
                }

                _links[i] = link;
            }
        }

        public float CompatabilityScore(GenomeNEAT genomeToCompare)
        {
            var disjointNumber = 0f;
            var excessNumber = 0f;
            var matchedNumber = 0f;
            var weightDifference = 0f;

            var genome1Index = 0;
            var genome2Index = 0;

            var genome1Size = _links.Count;
            var genome2Size = genomeToCompare._links.Count;
            while (genome1Index < genome1Size || genome2Index < genome2Size)
            {
                if (genome1Index == genome1Size)
                {
                    genome2Index++;
                    excessNumber++;
                    continue;
                }

                if (genome2Index == genome2Size)
                {
                    genome1Index++;
                    excessNumber++;
                    continue;
                }

                var link1 = _links[genome1Index];
                var link2 = genomeToCompare._links[genome2Index];
                if (link1.InnovationID == link2.InnovationID)
                {
                    genome1Index++;
                    genome2Index++;
                    matchedNumber++;
                    weightDifference = Math.Abs(link1.Weight - link2.Weight);
                }
                else if (link1.InnovationID < link2.InnovationID)
                {
                    disjointNumber++;
                    genome1Index++;
                }
                else
                {
                    disjointNumber++;
                    genome2Index++;
                }
            }

            var longestGenome = genome1Size > genome2Size ? genome1Size : genome2Size;
            //TODO: should be settable parameters
            const float disjointWeight = 1;
            const float excessWeight = 1;
            const float matchedWeight = 0.4f;

            return excessNumber * excessWeight / longestGenome + disjointNumber * disjointWeight / longestGenome +
                   weightDifference * matchedWeight / matchedNumber;
        }

        //TODO: test this code
        public void SortGenes()
        {
            _neurons.Sort((a, b) => a.SplitX.CompareTo(b.SplitX));
            _links.Sort((a, b) => a.InnovationID.CompareTo(b.InnovationID));
        }

        private bool DuplicateLink(int neuronIn, int neuronOut)
        {
            for (int i = 0; i < _links.Count; i++)
            {
                var currentLink = _links[i];
                if (currentLink.FromNeuron == neuronIn && currentLink.ToNeuron == neuronOut) return true;
            }

            return false;
        }

        private int GetElementPos(int neuronId)
        {
            for (int i = 0; i < _neurons.Count; i++)
            {
                var currentNeuron = _neurons[i];
                if (currentNeuron.Id == neuronId) return i;
            }

            return -1;
        }
    }

    public struct LinkGene
    {
        public readonly int InnovationID;
        public readonly int FromNeuron;
        public readonly int ToNeuron;
        public float Weight;
        public bool IsEnabled;
        public readonly bool IsRecurrent;

        public LinkGene(int innovationID, int fromNeuron, int toNeuron, float weight, bool isEnabled, bool isRecurrent)
        {
            InnovationID = innovationID;
            FromNeuron = fromNeuron;
            ToNeuron = toNeuron;
            Weight = weight;
            IsEnabled = isEnabled;
            IsRecurrent = isRecurrent;
        }
    }

    public struct NeuronGene
    {
        public readonly int Id;
        public readonly NeuronType NeuronType;
        public readonly float SplitX;

        // Might not be needed
        public bool IsRecurrent;

        // Not in the original implementation
        //public ActivationFunction ActivationFunction;
        //public float Bias;
        public NeuronGene(int id, NeuronType neuronType, float splitX, bool isRecurrent = false)
        {
            Id = id;
            NeuronType = neuronType;
            SplitX = splitX;
            IsRecurrent = isRecurrent;
        }
    }
}