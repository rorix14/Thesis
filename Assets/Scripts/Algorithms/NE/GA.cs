using UnityEngine;

namespace Algorithms.NE
{
    public class GA : ES
    {
        private readonly GAModel _gaNetworkModel;
        private readonly CrossoverInfo[] _crossoverInfos;
        private readonly float[] _mutationsVolume;

        private readonly int _tournamentSize;
        private readonly int _elitism;
        private readonly int[] _elitismIndexes;

        private readonly float _mutationMax;
        private readonly float _mutationMin;

        //Cashed variables
        private readonly float[] _elitismFitness;
        private readonly float[] _populationFitness;

        //TODO: model can't be null!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        public GA(GAModel networkModel, int numberOfActions, int batchSize, int elitism, int tournamentSize,
            float mutationMax = 0.10f,
            float mutationMin = 0.02f) : base(null, numberOfActions, batchSize)
        {
            _gaNetworkModel = networkModel;
            _populationFitness = new float[batchSize];
            _elitism = elitism;
            _tournamentSize = tournamentSize;
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

                    _elitismFitness[j] = individualFitness;
                    _elitismIndexes[j] = i;
                    break;
                }
            }

            _episodeRewardMean /= _batchSize;

            for (int i = _elitism; i < _batchSize; i++)
            {
                var crossoverInfo = new CrossoverInfo();

                var fitness1 = float.MinValue;
                var fitness2 = float.MinValue;
                for (int j = 0; j < _tournamentSize; j++)
                {
                    var individual = Random.Range(0, _batchSize);
                    var individualFitness = _populationFitness[individual];

                    if (fitness1 < individualFitness)
                    {
                        fitness1 = individualFitness;
                        crossoverInfo.Parent1 = individual;
                    }
                    else if (fitness2 < individualFitness)
                    {
                        fitness2 = individualFitness;
                        crossoverInfo.Parent2 = individual;
                    }

                    _crossoverInfos[i] = crossoverInfo;
                    
                    _mutationsVolume[i] = (fitness1 < _episodeRewardMean ? _mutationMax : _mutationMin) +
                                          (fitness2 < _episodeRewardMean ? _mutationMax : _mutationMin);
                }
            }

            for (int i = 0; i < _elitism; i++)
            {
                var parent = _elitismIndexes[i];
                _crossoverInfos[i] = new CrossoverInfo(parent, parent, 0);
                _elitismFitness[i] = float.MinValue;
            }
            
            _finishedIndividuals = 0;

            _gaNetworkModel.Update(_crossoverInfos, _mutationsVolume);
        }
    }
}