﻿using System.Windows;

namespace LiveCameraSample
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private void Application_Exit(object sender, ExitEventArgs e)
        {
            LiveCameraSample.Properties.Settings.Default.Save();
        }
    }
}
