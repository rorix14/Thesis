using NN;
using UnityEngine;

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
        }

        public override void Train()
        {
            for (int i = 0; i < _batchSize; i++)
            {
                var individualFitness = _episodeRewards[i];
                _episodeRewardMean += individualFitness;
                _populationFitness[i] = individualFitness;
                _episodeRewards[i] = 0f;
                _completedAgents[i] = false;

                for (int j = 0; j < _elitism; j++)
                {
                    if (_elitismFitness[j] >= individualFitness) continue;
                    
                    for (int k = _elitism - 1; k > j; k--)
                    {
                        _elitismFitness[k] = _elitismFitness[k - 1];
                        _elitismIndexes[k] =  _elitismIndexes[k - 1];
                    }
                        
                    _elitismFitness[j] = individualFitness;
                    _elitismIndexes[j] = i;
                    break;
                }
            }

            _episodeRewardMean /= _batchSize;

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

                    _crossoverInfos[i] = crossoverInfo;

                    var mutationVolume = (fitness1 < _episodeRewardMean ? _mutationMax : _mutationMin) +
                                         (fitness2 < _episodeRewardMean ? _mutationMax : _mutationMin);
                    _mutationsVolume[i] = mutationVolume;

                    if (i + 1 >= _batchSize) continue;

                    _crossoverInfos[i + 1] = new CrossoverInfo(crossoverInfo.Parent2, crossoverInfo.Parent1, 0);
                    _mutationsVolume[i + 1] = mutationVolume;
                }
            }

            for (int i = 0; i < _elitism; i++)
            {
                var parent = _elitismIndexes[i];
                _crossoverInfos[i] = new CrossoverInfo(parent, parent, 0);
                _elitismFitness[i] = float.MinValue;
            }

            _finishedIndividuals = 0;

            _gaModel.Update(_crossoverInfos, _mutationsVolume);
        }
    }
}