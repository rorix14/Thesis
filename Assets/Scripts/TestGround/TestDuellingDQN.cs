using Algorithms.RL;
using NN;
using UnityEngine;

namespace TestGround
{
    public class TestDuellingDQN : TestDQN
    {
        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var inputLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.ReLu, Instantiate(shader), true),
            };
            var inputModel = new NetworkModel(inputLayers, new NoLoss(Instantiate(shader)));

            var valueLayers = new NetworkLayer[]
            {
                new NetworkLayer(128, 128, ActivationFunction.ReLu, Instantiate(shader)),
                new NetworkLayer(128, 1, ActivationFunction.Linear, Instantiate(shader)),
            };
            var valueModel = new NetworkModel(valueLayers, new NoLoss(Instantiate(shader)));

            var advantageLayers = new NetworkLayer[]
            {
                new NetworkLayer(128, 128, ActivationFunction.ReLu, Instantiate(shader)),
                new NetworkLayer(128, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader))
            };
            var advantageModel = new NetworkModel(advantageLayers, new NoLoss(Instantiate(shader)));

            DuellingNetwork updateModel = new DuellingNetwork(inputModel, valueModel, advantageModel);

            // target creation
            var inputTargetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.ReLu, Instantiate(shader), true),
            };
            var inputTargetModel = new NetworkModel(inputTargetLayers, new NoLoss(Instantiate(shader)));

            var valueTargetLayers = new NetworkLayer[]
            {
                new NetworkLayer(128, 128, ActivationFunction.ReLu, Instantiate(shader)),
                new NetworkLayer(128, 1, ActivationFunction.Linear, Instantiate(shader)),
            };
            var valueTargetModel = new NetworkModel(valueTargetLayers, new NoLoss(Instantiate(shader)));

            var advantageTargetLayers = new NetworkLayer[]
            {
                new NetworkLayer(128, 128, ActivationFunction.ReLu, Instantiate(shader)),
                new NetworkLayer(128, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader))
            };
            var advantageTargetModel = new NetworkModel(advantageTargetLayers, new NoLoss(Instantiate(shader)));

            DuellingNetwork targetModel = new DuellingNetwork(inputTargetModel, valueTargetModel, advantageTargetModel);

            _DQN = new DuellingDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}