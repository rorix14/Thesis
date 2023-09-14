using System;
using System.Collections.Generic;
using NN;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Algorithms.NE
{
    public class GA : ES
    {
        private readonly GAModel _gaModel;
        private readonly CrossoverInfo[] _crossoverInfos;
        private readonly float[] _mutationsVolume;

        private readonly int _tournamentSize;
        private readonly int _elitism;
        private readonly int[] _elitismIndexes;

        private readonly float _mutationMax;
        private readonly float _mutationMin;

        //Cashed variables
        private readonly int[] _tournamentIndexes;
        private readonly float[] _elitismFitness;
        private readonly float[] _populationFitness;

        //NS
        private readonly List<Vector2> _archive;
        private float _noveltyThreshold;
        private readonly float _noveltyThresholdMinValue;
        private int _timeout;
        private readonly int _neighboursToCheck;
        private readonly float[] _noveltyScores;

        private readonly List<float>[] _agentsArchiveDistances;

        public GA(NetworkModel networkModel, int numberOfActions, int batchSize, int elitism, int tournamentSize,
            float mutationMax = 0.10f,
            float mutationMin = 0.02f) : base(networkModel, numberOfActions, batchSize)
        {
            _gaModel = (GAModel)networkModel;
            _crossoverInfos = new CrossoverInfo[batchSize];
            _mutationsVolume = new float[batchSize];
            _populationFitness = new float[batchSize];
            _elitism = elitism;
            _tournamentSize = tournamentSize;
            _tournamentIndexes = new int[tournamentSize];
            _elitismIndexes = new int[elitism];
            _elitismFitness = new float[elitism];
            _mutationMax = mutationMax;
            _mutationMin = mutationMin;

            for (int i = 0; i < elitism; i++)
            {
                _elitismFitness[i] = float.MinValue;
            }

            //NS 
            _archive = new List<Vector2>(batchSize);
            // one third of the max possible distance
            _noveltyThreshold = 5f;
            _noveltyThresholdMinValue = 1f;
            _timeout = 0;
            _neighboursToCheck = 10;
            _noveltyScores = new float[batchSize];
            _agentsArchiveDistances = new List<float>[batchSize];
        }

        public void DoNoveltySearch(Vector2[] agentsFinalPositions)
        {
            //TODO: needs to work if we want to do multiple episodes before training, example Moving Goal scene
            var addedToArchive = 0;
            for (int i = 0; i < _batchSize; i++)
            {
                var agentPosition = agentsFinalPositions[i];
                var archiveSize = _archive.Count;
                _agentsArchiveDistances[i] = new List<float>(archiveSize + _batchSize);

                var minDistance = float.MaxValue;
                for (int j = 0; j < archiveSize; j++)
                {
                    var archivePos = _archive[j];
                    var xDist = agentPosition.x - archivePos.x;
                    var yDist = agentPosition.y - archivePos.y;
                    var currentDistance = (float)Math.Sqrt(xDist * xDist + yDist * yDist);
                    _agentsArchiveDistances[i].Add(currentDistance);

                    if (minDistance < currentDistance) continue;

                    minDistance = currentDistance;
                }

                if (minDistance > _noveltyThreshold)
                {
                    _archive.Add(agentPosition);
                    addedToArchive++;
                }

                for (int j = 0; j < _batchSize; j++)
                {
                    var currentPos = agentsFinalPositions[j];
                    var xDist = agentPosition.x - currentPos.x;
                    var yDist = agentPosition.y - currentPos.y;
                    var currentDistance = (float)Math.Sqrt(xDist * xDist + yDist * yDist);
                    _agentsArchiveDistances[i].Add(currentDistance);
                }

                _agentsArchiveDistances[i].Sort();

                var distancesSum = 0f;
                for (int j = 1; j < _neighboursToCheck; j++)
                {
                    distancesSum += _agentsArchiveDistances[i][j];
                }

                _noveltyScores[i] = (distancesSum + 1) / (_neighboursToCheck - 1);
            }

            if (addedToArchive == 0)
            {
                _timeout++;
                if (_timeout <= 10) return;

                _timeout = 0;
                _noveltyThreshold *= 0.9f;
                if (_noveltyThreshold < _noveltyThresholdMinValue)
                {
                    _noveltyThreshold = _noveltyThresholdMinValue;
                }
            }
            else
            {
                _timeout = 0;
                if (addedToArchive > 4)
                {
                    _noveltyThreshold *= 1.2f;
                }
            }
        }

        public override void Train()
        {
            _episodeRewardMean = 0f;
            _episodeBestReward = float.MinValue;
            for (int i = 0; i < _batchSize; i++)
            {
                var individualFitness = _episodeRewards[i];
                _episodeRewardMean += individualFitness;
                _episodeBestReward = individualFitness > _episodeBestReward ? individualFitness : _episodeBestReward;
                _episodeRewards[i] = 0f;
                _completedAgents[i] = false;

                //TODO: 8 is a magic number, this should be done differently, for example: _noveltyBonus * _noveltyScores[i] + individualFitness
                //individualFitness = (individualFitness + 8) * _noveltyScores[i];
                _populationFitness[i] = individualFitness;

                for (int j = 0; j < _elitism; j++)
                {
                    if (_elitismFitness[j] >= individualFitness) continue;

                    for (int k = _elitism - 1; k > j; k--)
                    {
                        _elitismFitness[k] = _elitismFitness[k - 1];
                        _elitismIndexes[k] = _elitismIndexes[k - 1];
                    }

                    _elitismFitness[j] = individualFitness;
                    _elitismIndexes[j] = i;
                    break;
                }
            }

            _episodeRewardMean /= _batchSize;
            _finishedIndividuals = 0;

            Crossover();
            _gaModel.Update(_crossoverInfos, _mutationsVolume);
        }

        private void Crossover()
        {
            //TODO: This could be done using multithreading, although it might not be worth it    
            for (int i = _elitism; i < _batchSize; i += 2)
            {
                var crossoverInfo = new CrossoverInfo();

                var fitness1 = float.MinValue;
                var fitness2 = float.MinValue;

                var tournamentIteration = 0;
                while (tournamentIteration < _tournamentSize)
                {
                    _tournamentIndexes[tournamentIteration] = -1;
                    var individual = Random.Range(0, _batchSize);
                    var hasIndex = false;

                    for (int j = 0; j < tournamentIteration + 1; j++)
                    {
                        if (_tournamentIndexes[j] != individual) continue;

                        hasIndex = true;
                        break;
                    }

                    if (hasIndex) continue;

                    _tournamentIndexes[tournamentIteration] = individual;
                    tournamentIteration++;

                    var individualFitness = _populationFitness[individual];
                    if (fitness1 < individualFitness)
                    {
                        fitness2 = fitness1;
                        crossoverInfo.Parent2 = crossoverInfo.Parent1;
                        fitness1 = individualFitness;
                        crossoverInfo.Parent1 = individual;
                    }
                    else if (fitness2 < individualFitness)
                    {
                        fitness2 = individualFitness;
                        crossoverInfo.Parent2 = individual;
                    }
                }

                _crossoverInfos[i] = crossoverInfo;

                var mutationVolume = (fitness1 < _episodeRewardMean ? _mutationMax : _mutationMin) +
                                     (fitness2 < _episodeRewardMean ? _mutationMax : _mutationMin);
                _mutationsVolume[i] = mutationVolume;

                if (i + 1 >= _batchSize) continue;

                _crossoverInfos[i + 1] = new CrossoverInfo(crossoverInfo.Parent2, crossoverInfo.Parent1, 0);
                _mutationsVolume[i + 1] = mutationVolume;
            }

            for (int i = 0; i < _elitism; i++)
            {
                var parent = _elitismIndexes[i];
                _crossoverInfos[i] = new CrossoverInfo(parent, parent, 0);
                _elitismFitness[i] = float.MinValue;
            }
        }

        //Only use if the softmax activation function is used in the output layer
        // public override int[] SamplePopulationActions(float[,] states)
        // {
        //     _modelPredictions = _networkModel.Predict(states);
        //     
        //     for (int i = 0; i < _batchSize; i++)
        //     {
        //         if (_completedAgents[i])
        //         {
        //             _sampledActions[i] = -1;
        //             continue;
        //         }
        //
        //         var randomPoint = Random.value;
        //         _sampledActions[i] = 0;
        //         for (int j = 0; j < _numberOfActions; j++)
        //         {
        //             var probability = _modelPredictions[i, j];
        //             if (randomPoint < probability)
        //             {
        //                 _sampledActions[i] = j;
        //                 break;
        //             }
        //             
        //             randomPoint -= probability;
        //         }
        //     }
        //
        //     return _sampledActions;
        // }
    }
}