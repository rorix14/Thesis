using System;
using System.Collections.Generic;
using ActivationFunction = NN.ActivationFunction;
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
        private readonly int _genomeId;
        private readonly List<LinkGene> _links;
        private readonly List<NeuronGene> _neurons;
        private readonly int _inputNumber;
        private readonly int _outputNumber;

        public float Fitness;
        private float _adjustedFitness;
        private float _amountToSpawn;

        private List<float>[] _neuronsWeights;
        private List<int>[] _neuronsOutputIndexes;
        private float[] _neuronOutputs;

        private readonly float _disjointWeight;
        private readonly float _excessWeight;
        private readonly float _matchedWeight;

        // cashed variables
        private float[] _networkOutputs;
        private int _outputNeuronStart;

        public int GenomeId => _genomeId;
        public int GenomeSize => _links.Count;

        public float AdjustedFitness
        {
            get => _adjustedFitness;
            set => _adjustedFitness = value;
        }

        public LinkGene GetLinkGene(int index) => _links[index];
        public NeuronGene GetNeuronById(int id) => _neurons[GetElementPos(id)];

        public GenomeNEAT(int genomeId, int inputNumber, int outputNumber, float disjointWeight, float excessWeight,
            float matchedWeight)
        {
            _genomeId = genomeId;
            _inputNumber = inputNumber;
            _outputNumber = outputNumber;

            _disjointWeight = disjointWeight;
            _excessWeight = excessWeight;
            _matchedWeight = matchedWeight;

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
                    _links.Add(new LinkGene(innovationNumber++, inputNeuronId, _neurons[inputNumber + j].Id,
                        0.005f * Random.Range(-4f, 4f), true, false));
                }
            }
        }

        public GenomeNEAT(int genomeId, List<LinkGene> links, List<NeuronGene> neurons, int inputNumber,
            int outputNumber, float disjointWeight, float excessWeight, float matchedWeight)
        {
            _genomeId = genomeId;
            _links = links;
            _neurons = neurons;
            _inputNumber = inputNumber;
            _outputNumber = outputNumber;

            _disjointWeight = disjointWeight;
            _excessWeight = excessWeight;
            _matchedWeight = matchedWeight;
        }

        public GenomeNEAT(int genomeId, GenomeNEAT genomeToCopy)
        {
            _genomeId = genomeId;
            _links = new List<LinkGene>(genomeToCopy._links);
            _neurons = new List<NeuronGene>(genomeToCopy._neurons);

            _inputNumber = genomeToCopy._inputNumber;
            _outputNumber = genomeToCopy._outputNumber;

            _disjointWeight = genomeToCopy._disjointWeight;
            _excessWeight = genomeToCopy._excessWeight;
            _matchedWeight = genomeToCopy._matchedWeight;
        }

        public void CreatePhenotype()
        {
            var phenotypeLenght = _neurons.Count - _inputNumber;
            _neuronOutputs = new float[_neurons.Count];
            _neuronsWeights = new List<float>[phenotypeLenght];
            _neuronsOutputIndexes = new List<int>[phenotypeLenght];
            for (int i = 0; i < phenotypeLenght; i++)
            {
                _neuronsWeights[i] = new List<float>();
                _neuronsOutputIndexes[i] = new List<int>();
            }

            for (int i = 0; i < _links.Count; i++)
            {
                var link = _links[i];
                if (!link.IsEnabled) continue;

                var toNeuronIndex = GetElementPos(link.ToNeuron);
                _neuronsWeights[toNeuronIndex - _inputNumber].Add(link.Weight);
                _neuronsOutputIndexes[toNeuronIndex - _inputNumber].Add(GetElementPos(link.FromNeuron));
            }

            _networkOutputs = new float[_outputNumber];
            _outputNeuronStart = phenotypeLenght - _outputNumber;
        }

        public float[] Forward(float[] input, ActivationFunction activationFunction)
        {
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

                if (i < _outputNeuronStart)
                {
                    if (activationFunction == ActivationFunction.Tanh)
                    {
                        var expPos = (float)Math.Exp(sum);
                        var expNeg = (float)Math.Exp(-sum);
                        sum = (expPos - expNeg) / (expPos + expNeg);
                    }
                    else
                    {
                        sum = sum < 0.0f ? 0.0f : sum;
                    }
                }
                else
                {
                    _networkOutputs[i - _outputNeuronStart] = sum;
                }

                _neuronOutputs[_inputNumber + i] = sum;
            }

            return _networkOutputs;
        }

        public void AddLink(float mutationRate, float recurrentChance, InnovationNEAT innovation, int numTrysToFindLoop,
            int numTrysToAddLink, float[] weightsRandomBuffer)
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
                    var neuronId = _neurons[neuronPos].Id;

                    if (DuplicateLink(neuronId, neuronId)) continue;

                    neuron1Id = neuronId;
                    neuron2Id = neuronId;

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

                    if (!(DuplicateLink(neuron1Id, neuron2Id) || Math.Abs(neuron1.SplitX - neuron2.SplitX) <= 0f))
                    {
                        if (neuron1.SplitX > neuron2.SplitX)
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
            }

            _links.Add(new LinkGene(linkId, neuron1Id, neuron2Id,
                weightsRandomBuffer[Random.Range(0, weightsRandomBuffer.Length)], true, isRecurrent));
        }

        public void AddNeuron(float mutationRate, InnovationNEAT innovation)
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

            int link1Id;
            int link2Id;
            var newNeuronId = innovation.CheckInnovation(fromNeuron, toNeuron, NeuronType.Hidden);

            if (newNeuronId >= 0 && GetElementPos(newNeuronId) < 0)
            {
                link1Id = innovation.CheckInnovation(fromNeuron, newNeuronId, NeuronType.None);
                link2Id = innovation.CheckInnovation(newNeuronId, toNeuron, NeuronType.None);
            }
            else
            {
                newNeuronId = innovation.CreateNewInnovation(fromNeuron, toNeuron, NeuronType.Hidden);
                link1Id = innovation.CreateNewInnovation(fromNeuron, newNeuronId, NeuronType.None);
                link2Id = innovation.CreateNewInnovation(newNeuronId, toNeuron, NeuronType.None);
            }

            _neurons.Add(new NeuronGene(newNeuronId, NeuronType.Hidden, newDepth));
            _links.Add(new LinkGene(link1Id, fromNeuron, newNeuronId, 1f, true, false));
            _links.Add(new LinkGene(link2Id, newNeuronId, toNeuron, originalWeight, true, false));
        }

        public void MutateWeights(float mutationRate, float weightReplaceRate, float[] weightsRandomBuffer,
            float[] noiseRandomBuffer)
        {
            if (Random.value > mutationRate) return;

            var bufferLenght = weightsRandomBuffer.Length;
            for (int i = 0; i < _links.Count; i++)
            {
                var link = _links[i];
                link.Weight = Random.value < weightReplaceRate
                    ? weightsRandomBuffer[Random.Range(0, bufferLenght)]
                    : link.Weight + noiseRandomBuffer[Random.Range(0, bufferLenght)];

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
            return excessNumber * _excessWeight / longestGenome + disjointNumber * _disjointWeight / longestGenome +
                   weightDifference * _matchedWeight / matchedNumber;
        }

        public void SortGenes()
        {
            _links.Sort((a, b) => a.InnovationID.CompareTo(b.InnovationID));

            _neurons.Sort((a, b) =>
            {
                if (a.NeuronType == NeuronType.Hidden)
                {
                    //TODO: This comparison is probably not needed
                    if (b.NeuronType == NeuronType.Hidden && Math.Abs(a.SplitX - b.SplitX) <= 0f)
                    {
                        return a.Id.CompareTo(b.Id);
                    }

                    return a.SplitX.CompareTo(b.SplitX);
                }
                else if (a.NeuronType == NeuronType.Output)
                {
                    if (b.NeuronType == NeuronType.Hidden || b.NeuronType == NeuronType.Input)
                    {
                        return a.SplitX.CompareTo(b.SplitX);
                    }

                    return a.Id.CompareTo(b.Id);
                }
                else if (a.NeuronType == NeuronType.Input)
                {
                    if (b.NeuronType == NeuronType.Input)
                    {
                        return a.Id.CompareTo(b.Id);
                    }

                    return a.SplitX.CompareTo(b.SplitX);
                }

                return 0;
            });
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
            //TODO: consider doing an else if where we check if we have an id of an output neuron, neuronCount - outputNumber
            var searchStart = 0;
            if (neuronId > _inputNumber)
            {
                searchStart = _inputNumber;
            }

            for (int i = searchStart; i < _neurons.Count; i++)
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

        // Not in the original implementation
        //public ActivationFunction ActivationFunction;
        //public float Bias;
        public NeuronGene(int id, NeuronType neuronType, float splitX)
        {
            Id = id;
            NeuronType = neuronType;
            SplitX = splitX;
        }
    }
}