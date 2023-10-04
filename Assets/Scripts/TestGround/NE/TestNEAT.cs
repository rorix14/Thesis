using Algorithms.NE.NEAT;
using UnityEngine;

namespace TestGround.NE
{
    public class TestNEAT : TestES
    {
        public override string GetDescription()
        {
            return "NEAT" + (noveltyRelevance > 0 ? "-NS" : "") + ", " + activationFunction + ", " + populationSize +
                   " population size, noise std " + noiseStandardDeviation + ", novelty relevance " + noveltyRelevance +
                   ", initialization std " + weightsInitStd;
        }

        protected override void Start()
        {
            //Random.InitState(42);

            _env.CreatePopulation(populationSize);
            _currentSates = _env.DistributedResetEnv();

            var neatModel = new NEATModel(populationSize, _env.GetObservationSize, _env.GetNumberOfActions,
                activationFunction, weightsInitStd, noiseStandardDeviation);
            _neModel = new NEAT(neatModel, _env.GetNumberOfActions, populationSize, noveltyRelevance);

            Time.timeScale = simulationSpeed;
        }
    }
}