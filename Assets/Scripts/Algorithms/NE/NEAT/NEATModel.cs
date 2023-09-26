using System;
using System.Collections.Generic;
using NN.CPU_Single;
using UnityEngine;
using ActivationFunction = NN.ActivationFunction;
using Random = UnityEngine.Random;

namespace Algorithms.NE.NEAT
{
    public class NEATModel
    {
        private readonly int _populationSize;
        private readonly int _inputNumber;
        private readonly int _outputNumber;

        private readonly int _numGensAllowedNoImprovement;
        private readonly float _addLinkRate;
        private readonly float _addRecurrentLinkRate;
        private readonly float _addNeuronRate;
        private readonly float _weightMutationRate;
        private readonly float _weightReplaceRate;
        private readonly float _crossOverRate;
        private readonly float _speciesCompatibilityThreshold;
        private readonly int _speciesOldThreshold;
        private readonly float _speciesOldPenalty;
        private readonly int _speciesYoungThreshold;
        private readonly float _speciesYoungBonus;
        private readonly float _compatabilityDisjointWeight;
        private readonly float _compatabilityExcessWeight;
        private readonly float _compatabilityMatchedWeight;

        private readonly float[] _weightsRandomBuffer;
        private readonly float[] _noiseRandomBuffer;

        private readonly InnovationNEAT _innovationDB;
        private readonly GenomeNEAT[] _populationGenomes;
        private readonly List<SpeciesNEAT> _species;

        private SpeciesNEAT _bestSpecie;

        private int _genomeCount;
        private int _speciesCount;

        private ActivationFunction _activationFunction;

        // cashed variables
        private readonly float[] _populationOutput;
        private readonly float[] _individualInput;

        public NEATModel(int populationSize, int inputNumber, int outputNumber, ActivationFunction activationFunction, float neuronWeightStd = 0.005f,
            float noiseStd = 0.01f, int numGensAllowedNoImprovement = 75, float addLinkRate = 0.15f,
            float addRecurrentLinkRate = 0.05f, float addNeuronRate = 0.1f, float weightMutationRate = 0.15f,
            float weightReplaceRate = 0.1f, float crossOverRate = 0.8f, float speciesCompatibilityThreshold = 0.26f,
            int speciesOldThreshold = 50, float speciesOldPenalty = 0.3f, int speciesYoungThreshold = 10,
            float speciesYoungBonus = 0.3f, float compatabilityDisjointWeight = 8.5f,
            float compatabilityExcessWeight = 8.5f, float compatabilityMatchedWeight = 4f)
        {
            _populationSize = populationSize;
            _inputNumber = inputNumber;
            _outputNumber = outputNumber;
            _numGensAllowedNoImprovement = numGensAllowedNoImprovement;
            _addLinkRate = addLinkRate;
            _addRecurrentLinkRate = addRecurrentLinkRate;
            _addNeuronRate = addNeuronRate;
            _weightMutationRate = weightMutationRate;
            _weightReplaceRate = weightReplaceRate;
            _crossOverRate = crossOverRate;
            _speciesCompatibilityThreshold = speciesCompatibilityThreshold;
            _speciesOldThreshold = speciesOldThreshold;
            _speciesOldPenalty = speciesOldPenalty;
            _speciesYoungThreshold = speciesYoungThreshold;
            _speciesYoungBonus = speciesYoungBonus;
            _compatabilityDisjointWeight = compatabilityDisjointWeight;
            _compatabilityExcessWeight = compatabilityExcessWeight;
            _compatabilityMatchedWeight = compatabilityMatchedWeight;
            _activationFunction = activationFunction;

            _genomeCount = 0;
            _speciesCount = 0;

            _populationOutput = new float[populationSize * outputNumber];
            _individualInput = new float[inputNumber];

            _weightsRandomBuffer = new float[10000000];
            _noiseRandomBuffer = new float[10000000];
            for (int i = 0; i < 10000000; i++)
            {
                _weightsRandomBuffer[i] = neuronWeightStd * Random.Range(-4f, 4f);
                _noiseRandomBuffer[i] = noiseStd * NnMath.RandomGaussian(-10, 10);
            }

            _species = new List<SpeciesNEAT>()
            {
                new SpeciesNEAT(
                    new GenomeNEAT(_genomeCount++, inputNumber, outputNumber, compatabilityDisjointWeight,
                        compatabilityExcessWeight, compatabilityMatchedWeight), _speciesCount++, _speciesYoungThreshold,
                    _speciesYoungBonus, _speciesOldThreshold, _speciesOldPenalty)
            };

            _bestSpecie = _species[0];

            _innovationDB = new InnovationNEAT(inputNumber, outputNumber);
            _populationGenomes = new GenomeNEAT[populationSize];

            for (int i = 0; i < populationSize; i++)
            {
                var newGenome = new GenomeNEAT(_genomeCount++, inputNumber, outputNumber, compatabilityDisjointWeight,
                    compatabilityExcessWeight, compatabilityMatchedWeight);
                newGenome.CreatePhenotype();
                _populationGenomes[i] = newGenome;
            }
        }

        public float[] Predict(float[,] input)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                for (int j = 0; j < _inputNumber; j++)
                {
                    _individualInput[j] = input[i, j];
                }

                var individualActions = _populationGenomes[i].Forward(_individualInput, _activationFunction);
                var individualStartIndex = _outputNumber * i;
                for (int j = 0; j < _outputNumber; j++)
                {
                    _populationOutput[individualStartIndex + j] = individualActions[j];
                }
            }

            return _populationOutput;
        }

        public void Update(float[] populationFitness)
        {
            var sepciess = _species.Count;
            for (int i = _species.Count - 1; i >= 0; i--)
            {
                var specie = _species[i];
                specie.ResetSpecieMembers();

                if (specie.BestFitnessSoFar < _bestSpecie.LeaderFitness)
                {
                    if (specie.GenerationsSinceImprovement > _numGensAllowedNoImprovement)
                    {
                        _species.Remove(specie);
                    }
                }
                else
                {
                    _bestSpecie = specie;
                }
            }

            for (int i = 0; i < _populationSize; i++)
            {
                _populationGenomes[i].Fitness = populationFitness[i];
            }
            
            if (_species.Count < sepciess)
            {
                Debug.Log("Removed = " + _species.Count);
            }

            var bestCompaScore = 0f;
            for (int i = 0; i < _populationSize; i++)
            {
                var currentGenome = _populationGenomes[i];
                // currentGenome.SortGenes();

                var hasSpecies = false;
                for (int j = 0; j < _species.Count; j++)
                {
                    var specie = _species[j];
                    var compatabilityScore = currentGenome.CompatabilityScore(specie.Leader);
                    if (compatabilityScore > bestCompaScore)
                        bestCompaScore = compatabilityScore;
                    
                    if (compatabilityScore > _speciesCompatibilityThreshold) continue;

                    specie.AddMember(currentGenome);
                    hasSpecies = true;
                    break;
                }

                if (!hasSpecies)
                {
                    _species.Add(new SpeciesNEAT(currentGenome, _speciesCount++, _speciesYoungThreshold,
                        _speciesYoungBonus, _speciesOldThreshold, _speciesOldPenalty));
                }
            }

            var populationAdjustedFitness = 0f;
            for (int i = 0; i < _species.Count; i++)
            {
                populationAdjustedFitness += _species[i].AdjustedFitness();
            }

            GenomeNEAT parent1;
            GenomeNEAT parent2;
            GenomeNEAT offSpring;
            var numSpawnedSoFar = 0;

            populationAdjustedFitness /= _populationSize;
            for (int i = 0; i < _species.Count; i++)
            {
                if (numSpawnedSoFar >= _populationSize) break;

                var specie = _species[i];
                var numToSpawn = Mathf.RoundToInt(specie.CalculateSpawnAmount(populationAdjustedFitness));
                while (numToSpawn > 0)
                {
                    if (specie.SpecieMemberCount == 1)
                    {
                        offSpring = specie.Leader;
                    }
                    else
                    {
                        parent1 = specie.GetRandomMember();
                        if (Random.value < _crossOverRate)
                        {
                            parent2 = specie.GetRandomMember(parent1);
                            offSpring = Crossover(parent1, parent2);
                        }
                        else
                        {
                            offSpring = new GenomeNEAT(_genomeCount++, parent1);
                        }

                        offSpring.AddNeuron(_addNeuronRate, _innovationDB);
                        offSpring.AddLink(_addLinkRate, _addRecurrentLinkRate, _innovationDB, 5, 5,
                            _weightsRandomBuffer);
                        offSpring.MutateWeights(_weightMutationRate, _weightReplaceRate, _weightsRandomBuffer,
                            _noiseRandomBuffer);
                    }

                    offSpring.SortGenes();
                    offSpring.CreatePhenotype();
                    _populationGenomes[numSpawnedSoFar] = offSpring;

                    numSpawnedSoFar++;
                    numToSpawn--;

                    if (numSpawnedSoFar < _populationSize) continue;

                    numToSpawn = 0;
                }
            }

            Debug.Log("Highest compa score: " + bestCompaScore);
            if (_species.Count > 1)
            {
                Debug.Log("Number of species = " + _species.Count);
            }

            //TODO: test this condition and then remove
            if (numSpawnedSoFar < _populationSize)
            {
                Debug.Log("Not enough offspring created.. try ceiling the spawn values");
            }
        }

        private GenomeNEAT Crossover(GenomeNEAT parent1, GenomeNEAT parent2)
        {
            GenomeNEAT bestParent;
            var parent1Size = parent1.GenomeSize;
            var parent2Size = parent2.GenomeSize;

            if (Math.Abs(parent1.Fitness - parent2.Fitness) <= 0.0f)
            {
                if (parent1Size == parent2Size)
                {
                    bestParent = Random.Range(0, 2) == 0 ? parent1 : parent2;
                }
                else
                {
                    bestParent = parent1Size < parent2Size ? parent1 : parent2;
                }
            }
            else
            {
                bestParent = parent1.Fitness > parent2.Fitness ? parent1 : parent2;
            }

            var offspringNeurons = new List<NeuronGene>(_inputNumber + _outputNumber);
            var offspringLinks = new List<LinkGene>(bestParent.GenomeSize);
            var neuronsIds = new List<int>(_inputNumber + _outputNumber);

            var parent1GeneIndex = 0;
            var parent2GeneIndex = 0;

            LinkGene parent1Gene;
            LinkGene parent2Gene;
            var selectedGene = new LinkGene();

            while (parent1GeneIndex < parent1Size || parent2GeneIndex < parent2Size)
            {
                if (parent1GeneIndex == parent1Size && parent2GeneIndex != parent2Size)
                {
                    if (bestParent == parent2)
                    {
                        selectedGene = parent2.GetLinkGene(parent2GeneIndex);
                    }

                    parent2GeneIndex++;
                }
                else if (parent2GeneIndex == parent2Size && parent1GeneIndex != parent1Size)
                {
                    if (bestParent == parent1)
                    {
                        selectedGene = parent1.GetLinkGene(parent1GeneIndex);
                    }

                    parent1GeneIndex++;
                }
                else if ((parent1Gene = parent1.GetLinkGene(parent1GeneIndex)).InnovationID <
                         parent2.GetLinkGene(parent2GeneIndex).InnovationID)
                {
                    if (bestParent == parent1)
                    {
                        selectedGene = parent1Gene;
                    }

                    parent1GeneIndex++;
                }
                else if ((parent2Gene = parent2.GetLinkGene(parent2GeneIndex)).InnovationID <
                         parent1.GetLinkGene(parent1GeneIndex).InnovationID)
                {
                    if (bestParent == parent2)
                    {
                        selectedGene = parent2Gene;
                    }

                    parent2GeneIndex++;
                }
                else
                {
                    selectedGene = Random.value < 0.5f
                        ? parent1.GetLinkGene(parent1GeneIndex)
                        : parent2.GetLinkGene(parent2GeneIndex);

                    parent1GeneIndex++;
                    parent2GeneIndex++;
                }

                if (offspringLinks.Count == 0)
                {
                    offspringLinks.Add(selectedGene);
                }
                else if (offspringLinks[offspringLinks.Count - 1].InnovationID != selectedGene.InnovationID)
                {
                    offspringLinks.Add(selectedGene);
                }

                var hasFromNeuronId = false;
                var hasToNeuronId = false;
                for (int i = 0; i < neuronsIds.Count; i++)
                {
                    var neuronId = neuronsIds[i];
                    if (neuronId == selectedGene.FromNeuron)
                    {
                        hasFromNeuronId = true;
                    }

                    if (neuronId == selectedGene.ToNeuron)
                    {
                        hasToNeuronId = true;
                    }

                    if (hasFromNeuronId && hasToNeuronId) break;
                }

                if (!hasFromNeuronId)
                {
                    neuronsIds.Add(selectedGene.FromNeuron);
                }

                if (!hasToNeuronId && selectedGene.FromNeuron != selectedGene.ToNeuron)
                {
                    neuronsIds.Add(selectedGene.ToNeuron);
                }
            }

            neuronsIds.Sort();
            for (int i = 0; i < neuronsIds.Count; i++)
            {
                offspringNeurons.Add(bestParent.GetNeuronById(neuronsIds[i]));
            }

            return new GenomeNEAT(_genomeCount++, offspringLinks, offspringNeurons, _inputNumber, _outputNumber,
                _compatabilityDisjointWeight, _compatabilityExcessWeight, _compatabilityMatchedWeight);
        }
    }
}