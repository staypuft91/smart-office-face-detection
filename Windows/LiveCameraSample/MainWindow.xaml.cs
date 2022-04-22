using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using LiveCameraSample.Properties;
using Microsoft.Azure.CognitiveServices.Vision.Face;
using Microsoft.Azure.CognitiveServices.Vision.Face.Models;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using VideoFrameAnalyzer;
using FaceAPI = Microsoft.Azure.CognitiveServices.Vision.Face;
using VisionAPI = Microsoft.Azure.CognitiveServices.Vision.ComputerVision;

namespace LiveCameraSample
{
    public partial class MainWindow : System.Windows.Window, IDisposable
    {
        private FaceClient faceClient = null;
        private readonly FrameGrabber<LiveCameraResult> grabber;
        private static readonly ImageEncodingParam[] s_jpegParams = {
            new ImageEncodingParam(ImwriteFlags.JpegQuality, 60)
        };
        private readonly CascadeClassifier localFaceDetector = new CascadeClassifier();
        private LiveCameraResult latestResultToDisplay = null;
        private DateTime startTime;
        private Settings settings;

        public MainWindow()
        {
            InitializeComponent();
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            settings = new Settings();

            // Clean leading/trailing spaces in API keys. 
            settings.FaceAPIKey = settings.FaceAPIKey.Trim();
            settings.VisionAPIKey = settings.VisionAPIKey.Trim();

            // Create API client
            faceClient = new FaceClient(new ApiKeyServiceClientCredentials(settings.FaceAPIKey))
            {
                Endpoint = settings.FaceAPIHost
            };

            // Create grabber. 
            grabber = new FrameGrabber<LiveCameraResult>();
            grabber.AnalysisFunction = FacesAnalysisFunction;

            // Set up a listener for when the client receives a new frame.
            grabber.NewFrameProvided += (s, e) =>
            {
                // The callback may occur on a different thread, so we must use the
                // MainWindow.Dispatcher when manipulating the UI. 
                Dispatcher.BeginInvoke((Action)(() =>
                {
                    // Display the image in the left pane.
                    LeftImage.Source = e.Frame.Image.ToBitmapSource();
                }));

                // See if auto-stop should be triggered. 
                if (settings.AutoStopEnabled && (DateTime.Now - startTime) > settings.AutoStopTime)
                {
                    grabber.StopProcessingAsync().GetAwaiter().GetResult();
                }
            };

            // Set up a listener for when the client receives a new result from an API call. 
            grabber.NewResultAvailable += (s, e) =>
            {
                Dispatcher.BeginInvoke((Action)(() =>
                {
                    if (e.TimedOut)
                    {
                        MessageArea.Text = "API call timed out.";
                    }
                    else if (e.Exception != null)
                    {
                        string apiName = "";
                        string message = e.Exception.Message;
                        var faceEx = e.Exception as APIErrorException;
                        var visionEx = e.Exception as VisionAPI.Models.ComputerVisionErrorException;
                        if (faceEx != null)
                        {
                            apiName = "Face";
                            message = faceEx.Message;
                        }
                        else if (visionEx != null)
                        {
                            apiName = "Computer Vision";
                            message = visionEx.Message;
                        }
                        MessageArea.Text = string.Format("{0} API call failed on frame {1}. Exception: {2}", apiName, e.Frame.Metadata.Index, message);
                    }
                    else
                    {
                        latestResultToDisplay = e.Analysis;
                        RightImage.Source = VisualizeResult(e.Frame);
                    }
                }));
            };

            // Create local face detector. 
            localFaceDetector.Load("Data/haarcascade_frontalface_alt2.xml");
        }


        private async void CreateAndTrainPersonGroup(object sender, RoutedEventArgs e)
        {
            // Create a dictionary for all your images, grouping similar ones under the same key.
            Dictionary<string, string[]> personDictionary = new Dictionary<string, string[]>
            {
                { "Cristian Rodriguez", new[] { "user-1-pic-1.png", "user-1-pic-2.jpg" } },
                { "Guillem Bonafonte", new[] { "user-2-pic-1.jpg", "user-2-pic-2.png" } }
            };

            // Check if person group exists
            var personGroup = await faceClient.PersonGroup.GetAsync(settings.PersonGroupId);
            if (personGroup == null)
            {
                await faceClient.PersonGroup.CreateAsync(settings.PersonGroupId, settings.PersonGroupId, recognitionModel: RecognitionModel.Recognition02);
            }

            // The similar faces will be grouped into a single person group person.
            foreach (var user in personDictionary)
            {
                await Task.Delay(250);
                Person person = await faceClient.PersonGroupPerson.CreateAsync(personGroupId: settings.PersonGroupId, name: user.Key);

                // Add face to the person group person.
                foreach (var userImageFileName in personDictionary[user.Key])
                {
                    // Create the file.
                    using (FileStream imageFileStream = File.OpenRead($"{settings.LocalUrl}/{userImageFileName}"))
                    {
                        PersistedFace face = await faceClient.PersonGroupPerson.AddFaceFromStreamAsync(settings.PersonGroupId, person.PersonId,
                            imageFileStream);
                    }
                }
            }

            // Start to train the person group
            await faceClient.PersonGroup.TrainAsync(settings.PersonGroupId);

            // Wait until the training is completed
            while (true)
            {
                await Task.Delay(5000);
                var trainingStatus = await faceClient.PersonGroup.GetTrainingStatusAsync(settings.PersonGroupId);
                if (trainingStatus.Status == TrainingStatusType.Succeeded) { break; }
            }
        }

        private async Task<LiveCameraResult> FacesAnalysisFunction(VideoFrame frame)
        {
            // Encode image. 
            var jpg = frame.Image.ToMemoryStream(".jpg", s_jpegParams);
            // Submit image to API. 
            var attrs = new List<FaceAttributeType> {
                FaceAttributeType.Age,
                FaceAttributeType.Gender,
                FaceAttributeType.HeadPose
            };
            var faces = await faceClient.Face.DetectWithStreamAsync(jpg, returnFaceAttributes: attrs);
            // Count the API call. 
            settings.FaceAPICallCount++;
            // Output. 
            return new LiveCameraResult { Faces = faces.ToArray() };
        }

        private BitmapSource VisualizeResult(VideoFrame frame)
        {
            // Draw any results on top of the image. 
            BitmapSource visImage = frame.Image.ToBitmapSource();

            var result = latestResultToDisplay;

            if (result != null)
            {
                // See if we have local face detections for this image.
                var clientFaces = (OpenCvSharp.Rect[])frame.UserData;
                if (clientFaces != null && result.Faces != null)
                {
                    // If so, then the analysis results might be from an older frame. We need to match
                    // the client-side face detections (computed on this frame) with the analysis
                    // results (computed on the older frame) that we want to display. 
                    MatchAndReplaceFaceRectangles(result.Faces, clientFaces);
                }

                visImage = Visualization.DrawFaces(visImage, result.Faces);
                visImage = Visualization.DrawTags(visImage, result.Tags);
            }

            return visImage;
        }

        private void LoadCameraList(object sender, RoutedEventArgs e)
        {
            int numCameras = grabber.GetNumCameras();

            if (numCameras == 0)
            {
                MessageArea.Text = "No cameras found!";
            }

            var comboBox = sender as ComboBox;
            comboBox.ItemsSource = Enumerable.Range(0, numCameras).Select(i => string.Format("Camera {0}", i + 1));
            comboBox.SelectedIndex = 0;
        }

        private async void StartCamera(object sender, RoutedEventArgs e)
        {
            if (!CameraList.HasItems)
            {
                MessageArea.Text = "No cameras found; cannot start processing";
                return;
            }

            // How often to analyze. 
            grabber.TriggerAnalysisOnInterval(settings.AnalysisInterval);

            // Reset message. 
            MessageArea.Text = "";

            // Record start time, for auto-stop
            startTime = DateTime.Now;

            await grabber.StartProcessingCameraAsync(CameraList.SelectedIndex);
        }

        private async void StopCamera(object sender, RoutedEventArgs e) => await grabber.StopProcessingAsync();

        private void SettingsVisibility(object sender, RoutedEventArgs e) => SettingsPanel.Visibility = 1 - SettingsPanel.Visibility;

        private void SaveSettings(object sender, RoutedEventArgs e)
        {
            SettingsPanel.Visibility = Visibility.Hidden;
            Properties.Settings.Default.Save();
        }

        private void Hyperlink_RequestNavigate(object sender, RequestNavigateEventArgs e)
        {
            Process.Start(new ProcessStartInfo(e.Uri.AbsoluteUri));
            e.Handled = true;
        }

        private void MatchAndReplaceFaceRectangles(FaceAPI.Models.DetectedFace[] faces, OpenCvSharp.Rect[] clientRects)
        {
            // Use a simple heuristic for matching the client-side faces to the faces in the
            // results. Just sort both lists left-to-right, and assume a 1:1 correspondence. 

            // Sort the faces left-to-right. 
            var sortedResultFaces = faces
                .OrderBy(f => f.FaceRectangle.Left + 0.5 * f.FaceRectangle.Width)
                .ToArray();

            // Sort the clientRects left-to-right.
            var sortedClientRects = clientRects
                .OrderBy(r => r.Left + 0.5 * r.Width)
                .ToArray();

            // Assume that the sorted lists now corrrespond directly. We can simply update the
            // FaceRectangles in sortedResultFaces, because they refer to the same underlying
            // objects as the input "faces" array. 
            for (int i = 0; i < Math.Min(faces.Length, clientRects.Length); i++)
            {
                // convert from OpenCvSharp rectangles
                OpenCvSharp.Rect r = sortedClientRects[i];
                sortedResultFaces[i].FaceRectangle = new FaceRectangle { Left = r.Left, Top = r.Top, Width = r.Width, Height = r.Height };
            }
        }

        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    grabber?.Dispose();
                    faceClient?.Dispose();
                    localFaceDetector?.Dispose();
                }

                disposedValue = true;
            }
        }

        public void Dispose() => Dispose(true);
    }
}
