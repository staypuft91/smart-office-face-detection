using Microsoft.Azure.CognitiveServices.Vision.Face.Models;
using Microsoft.Azure.CognitiveServices.Vision.ComputerVision.Models;

namespace LiveCameraSample
{
    // Class to hold all possible result types. 
    public class LiveCameraResult
    {
        public DetectedFace[] Faces { get; set; } = null;
        public ImageTag[] Tags { get; set; } = null;
    }
}
