using Stealth_Game;
using UnityEngine;

namespace Gym
{
    public class ImageStealthGameEnv : StealthGameEnv
    {
        [SerializeField] private Camera captureCamera;
        [SerializeField] private int imageWithHeight; // should be 30 or 36
        [SerializeField] private ComputeShader imageProcessorCs;

        // Image converter variables
        private ComputeShader _imageShader;
        private int _imageToMatrixKernel;
        private int _threadGroupsX;
        private ComputeBuffer _imageOutputBuffer;

        // cashed variables
        private RenderTexture _rt;
        private Texture2D _envStateTexture;
        private Rect _imageView;


        protected override void Start()
        {
            if (_envStarted) return;

            _envStarted = true;

            // ObservationLenght = imageWithHeight * imageWithHeight * 3;
            ObservationLenght = imageWithHeight * imageWithHeight;

            _imageShader = Instantiate(imageProcessorCs);

            // _imageToMatrixKernel = _imageShader.FindKernel("image_to_matrix");
            _imageToMatrixKernel = _imageShader.FindKernel("image_to_matrix_grayscale");

            _imageShader.SetInt("image_width", imageWithHeight);
            _imageShader.SetInt("image_height", imageWithHeight);
            _imageShader.SetInt("image_size", imageWithHeight * imageWithHeight);

            _imageShader.GetKernelThreadGroupSizes(_imageToMatrixKernel, out var x, out _, out _);
            _threadGroupsX = Mathf.CeilToInt(imageWithHeight / (float)x);

            _imageOutputBuffer = new ComputeBuffer(imageWithHeight * imageWithHeight * 3, sizeof(float));
            _imageShader.SetBuffer(_imageToMatrixKernel, "image_mat", _imageOutputBuffer);

            _rt = new RenderTexture(imageWithHeight, imageWithHeight, 24);

            _envStateTexture = new Texture2D(imageWithHeight, imageWithHeight, TextureFormat.RGB24, false, true);
            _imageShader.SetTexture(_imageToMatrixKernel, "input_texture", _envStateTexture);

            _imageView = new Rect(0, 0, imageWithHeight, imageWithHeight);
        }

        public override StepInfo Step(int actionIndex)
        {
            var action = ActionLookup[actionIndex];
            var observation = new float[ObservationLenght];
            var stepInfo = new StepInfo(observation, passiveReward, EpisodeLengthIndex > episodeLength);

            if (action.y != 0)
            {
                action.y = 0;
                if (_player.IterableObjects.Count > 0)
                {
                    var enemyToRemove = _player.IterableObjects[0].GetComponent<EnemyAgent>();
                    if (enemyToRemove)
                    {
                        enemyToRemove.KillAgent();
                        _player.IterableObjects.RemoveAt(0);
                        stepInfo.Reward = assassinateReward;
                    }
                }
            }

            _player.MovePlayer(action);

            var playerPosition = _player.transform.position;
            for (var i = 0; i < _enemyCount; i++)
            {
                var enemy = _enemies[i];
                enemy.UpdateEnemy();

                if (!enemy.CanSeeTarget(playerPosition)) continue;

                stepInfo.Done = true;
                stepInfo.Reward = spottedReward;
            }

            if (_player.GoalReached)
            {
                stepInfo.Done = true;
                stepInfo.Reward = goalReachedReward;
            }

            captureCamera.targetTexture = _rt;
            captureCamera.Render();

            RenderTexture.active = _rt;
            _envStateTexture.ReadPixels(_imageView, 0, 0);
            _envStateTexture.Apply();

            _imageShader.Dispatch(_imageToMatrixKernel, _threadGroupsX, _threadGroupsX, 1);
            _imageOutputBuffer.GetData(observation);

            RenderTexture.active = null;
            captureCamera.targetTexture = null;

            // if (EpisodeLengthIndex == 300) FillAndSaveImages(observation, "image_reset_" + test);

            EpisodeLengthIndex++;

            return stepInfo;
        }

        private int test = 0;

        public override float[] ResetEnv()
        {
            if (!_envStarted)
            {
                Start();
            }
            else
            {
                base.ResetEnv();
            }

            Physics.SyncTransforms();

            for (var i = 0; i < _enemyCount; i++)
            {
                _enemies[i].UpdateEnemy();
            }

            _resetObservation = new float[ObservationLenght];

            captureCamera.targetTexture = _rt;
            captureCamera.Render();

            RenderTexture.active = _rt;
            _envStateTexture.ReadPixels(_imageView, 0, 0);
            _envStateTexture.Apply();

            _imageShader.Dispatch(_imageToMatrixKernel, _threadGroupsX, _threadGroupsX, 1);
            _imageOutputBuffer.GetData(_resetObservation);

            RenderTexture.active = null;
            captureCamera.targetTexture = null;

            test++;

            return _resetObservation;
        }

        private void OnDestroy()
        {
            // if (imageCamera) { imageCamera.targetTexture = null; }
            // RenderTexture.active = null;

            _imageOutputBuffer?.Dispose();
        }

        //Only here for testing purposes, as it allows for us to visualize the captured game images 
        private void FillAndSaveImages(float[] matrix, string imageName)
        {
            var newImage = new Texture2D(imageWithHeight, imageWithHeight);
            for (int y = 0; y < imageWithHeight; y++)
            {
                for (int x = 0; x < imageWithHeight; x++)
                {
                    var matIndex = y * imageWithHeight + x;
                    // newImage.SetPixel(x, y,
                    //     new Color(matrix[matIndex], matrix[matIndex + imageWithHeight * imageWithHeight],
                    //         matrix[matIndex + imageWithHeight * imageWithHeight * 2]));
                    newImage.SetPixel(x, y, new Color(matrix[matIndex], matrix[matIndex], matrix[matIndex]));
                }
            }

            newImage.Apply();
            System.IO.File.WriteAllBytes(Application.dataPath + "/" + imageName + "_" + EpisodeLengthIndex + ".png",
                newImage.EncodeToPNG());
            //print(string.Concat("Image saved: ", Application.dataPath, "/", imageName, ".png"));
        }
    }
}