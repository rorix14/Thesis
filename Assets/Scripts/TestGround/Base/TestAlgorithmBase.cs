using System.Collections.Generic;
using UnityEngine;

namespace TestGround.Base
{
    public abstract class TestAlgorithmBase : MonoBehaviour
    {
        public bool IsFinished;
        public List<float> Rewards;
        public List<float> Loss;

        public abstract string GetDescription();
    }
}