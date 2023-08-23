using System;
using System.Collections.Generic;
using UnityEngine;
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

        private readonly InnovationNEAT _innovationDB;
        private readonly GenomeNEAT[] _populationGenomes;
        private readonly float[] _populationFitness;
        private readonly List<SpeciesNEAT> _species;

        private SpeciesNEAT _bestSpecie;

        private int _genomeCount;
        private int _speciesCount;

        public NEATModel(int populationSize, int inputNumber, int outputNumber, int numGensAllowedNoImprovement = 75,
            float addLinkRate = 0.07f, float addRecurrentLinkRate = 0.05f, float addNeuronRate = 0.03f,
            float weightMutationRate = 0.2f, float weightReplaceRate = 0.1f, float crossOverRate = 0.07f,
            float speciesCompatibilityThreshold = 0.26f, int speciesOldThreshold = 50, float speciesOldPenalty = 0.3f,
            int speciesYoungThreshold = 10, float speciesYoungBonus = 0.3f)
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

            _genomeCount = 0;
            _speciesCount = 0;

            _species = new List<SpeciesNEAT>()
            {
                new SpeciesNEAT(new GenomeNEAT(_genomeCount++, inputNumber, outputNumber), _speciesCount++,
                    _speciesYoungThreshold, _speciesYoungBonus, _speciesOldThreshold, _speciesOldPenalty)
            };

            _innovationDB = new InnovationNEAT(inputNumber, outputNumber);
            _populationGenomes = new GenomeNEAT[populationSize];
            _populationFitness = new float[populationSize];

            for (int i = 0; i < populationSize; i++)
            {
                _populationGenomes[i] = new GenomeNEAT(_genomeCount++, inputNumber, outputNumber);
            }
        }

        public float[] Predict(float[,] input)
        {
            //TODO: can be a class array
            var populationOutput = new float[_populationSize * _outputNumber];
            var individualInput = new float[_inputNumber];
            for (int i = 0; i < _populationSize; i++)
            {
                for (int j = 0; j < _inputNumber; j++)
                {
                    individualInput[i] = input[i, j];
                }
                
                var individualActions = _populationGenomes[i].Forward(individualInput);
                var individualStartIndex = _outputNumber * _populationSize;
                for (int j = 0; j < _outputNumber; j++)
                {
                    populationOutput[individualStartIndex + j] = individualActions[j];
                }
            }
            
            return populationOutput;
        }

        public void Update(float[] populationFitness)
        {
            //Destroy all phenotypes
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

            for (int i = 0; i < _populationSize; i++)
            {
                var currentGenome = _populationGenomes[i];

                //TODO: might not be necessary
                currentGenome.SortGenes();

                var hasSpecies = false;
                for (int j = 0; j < _species.Count; j++)
                {
                    var specie = _species[j];
                    if (currentGenome.CompatabilityScore(specie.Leader) > _speciesCompatibilityThreshold) continue;

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

            //TODO: must see if this works and outputs near the population size
            populationAdjustedFitness /= _populationSize;
            for (int i = 0; i < _species.Count; i++)
            {
                if (numSpawnedSoFar >= _populationSize) break;

                var specie = _species[i];
                var numToSpawn = Mathf.RoundToInt(specie.CalculateSpawnAmount(populationAdjustedFitness));
                //TODO: the book puts the species leader into the new population, witch is not done here
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
                        
                        offSpring.AddNeuron(_addNeuronRate, _innovationDB, 5);
                        offSpring.AddLink(_addLinkRate, _addRecurrentLinkRate, _innovationDB, 5, 5);
                        offSpring.MutateWeights(_weightMutationRate, _weightReplaceRate);
                    }
                    
                    //TODO: might not be necessary
                    offSpring.SortGenes();
                    
                    _populationGenomes[numSpawnedSoFar] = offSpring;
                    
                    numSpawnedSoFar++;
                    numToSpawn--;

                    if(numSpawnedSoFar < _populationSize) continue;
                    
                    numToSpawn = 0;
                }
            }

            //TODO: test this condition and then remove
            if (numSpawnedSoFar < _populationSize)
            {
                Debug.Log("Not enough offspring created.. try ceiling the spawn values");   
            }
        }

        public GenomeNEAT Crossover(GenomeNEAT parent1, GenomeNEAT parent2)
        {
            GenomeNEAT bestParent;
            var parent1Size = parent1.GenomeSize;
            var parent2Size = parent2.GenomeSize;

            if (Math.Abs(parent1.Fitness - parent2.Fitness) < 0.0f)
            {
                //TODO: could probably made shorter
                if (parent1Size == parent2Size)
                {
                    bestParent = Random.Range(0, 2) == 0 ? parent1 : parent2;
                }
                else if (parent1Size < parent2Size)
                {
                    bestParent = parent1;
                }
                else
                {
                    bestParent = parent2;
                }
            }
            else if (parent1.Fitness > parent2.Fitness)
            {
                bestParent = parent1;
            }
            else
            {
                bestParent = parent2;
            }

            var offspringNeurons = new List<NeuronGene>(_inputNumber + _outputNumber);
            var offspringLinks = new List<LinkGene>(bestParent.GenomeSize);
            var neuronsIds = new List<int>(_inputNumber + _outputNumber);

            var parent1GeneIndex = 0;
            var parent2GeneIndex = 0;

            var selectedGene = new LinkGene();
            //TODO: consider converting this condition into something more simple, as in Genome.GetCompatabilityScore
            while (!(parent1GeneIndex == parent1Size && parent2GeneIndex == parent2Size))
            {
                var parent1Gene = parent1.GetLinkGene(parent1GeneIndex);
                var parent2Gene = parent2.GetLinkGene(parent2GeneIndex);

                if (parent1GeneIndex == parent1Size && parent2GeneIndex != parent2Size)
                {
                    if (bestParent == parent2)
                    {
                        selectedGene = parent2Gene;
                    }

                    parent2GeneIndex++;
                }
                else if (parent2GeneIndex == parent2Size && parent1GeneIndex != parent1Size)
                {
                    if (bestParent == parent1)
                    {
                        selectedGene = parent1Gene;
                    }

                    parent1GeneIndex++;
                }
                else if (parent1Gene.InnovationID < parent2Gene.InnovationID)
                {
                    if (bestParent == parent1)
                    {
                        selectedGene = parent1Gene;
                    }

                    parent1GeneIndex++;
                }
                else if (parent2Gene.InnovationID < parent1Gene.InnovationID)
                {
                    if (bestParent == parent2)
                    {
                        selectedGene = parent2Gene;
                    }

                    parent2GeneIndex++;
                }
                else if (parent1Gene.InnovationID == parent2Gene.InnovationID)
                {
                    selectedGene = Random.value < 0.5f ? parent1Gene : parent2Gene;
                    parent1GeneIndex++;
                    parent2GeneIndex++;
                }

                //TODO: used for testing, remove once it is clear that the condition does not happen
                for (int i = 0; i < offspringLinks.Count; i++)
                {
                    if (offspringLinks[i].InnovationID != selectedGene.InnovationID) continue;
                    Debug.Log("Crossover: offspring innovation link already added");
                    break;
                }

                offspringLinks.Add(selectedGene);

                var hasFromNeuronId = false;
                var hasToNeuronId = false;
                for (int i = 0; i < neuronsIds.Count; i++)
                {
                    var neuronId = neuronsIds[i];
                    if (neuronId == selectedGene.FromNeuron)
                    {
                        hasFromNeuronId = true;
                    }
                    else if (neuronId == selectedGene.ToNeuron)
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

            return new GenomeNEAT(_genomeCount++, offspringLinks, offspringNeurons, _inputNumber, _outputNumber);
        }
    }
}