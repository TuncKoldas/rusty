use opencv::{
    core::{Size},
    highgui,
    imgproc,
    objdetect,
    prelude::*,
    videoio,
};

fn main() {
    let cascade_path = "haarcascade_frontalface_default.xml"; // Path to Haar cascade file
    let stream_url = "http://192.168.178.21:4747/video"; // Replace with your DroidCam IP

    // Load the cascade classifier
    let mut face_cascade =
        objdetect::CascadeClassifier::new(cascade_path).expect("Failed to load Haar cascade file");

    // Open the video stream from DroidCam
    let mut cap = videoio::VideoCapture::from_file(stream_url, videoio::CAP_ANY)
        .expect("Failed to open video stream");
    if !cap.is_opened().unwrap() {
        panic!("Unable to open camera stream");
    }

    // Create a window to display the video
    highgui::named_window("Face Detection", highgui::WINDOW_AUTOSIZE).unwrap();

    let mut frame = Mat::default();
    loop {
        // Capture frame-by-frame
        if !cap.read(&mut frame).unwrap() || frame.empty() {
            eprintln!("Failed to capture frame");
            break;
        }

        // Convert to grayscale for face detection
        let mut gray_frame = Mat::default();
        imgproc::cvt_color(&frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0).unwrap();

        // Equalize histogram
        let mut equalized_frame = Mat::default();
        imgproc::equalize_hist(&gray_frame, &mut equalized_frame).unwrap();
        gray_frame = equalized_frame;

        // Detect faces
        let mut faces = opencv::core::Vector::<opencv::core::Rect>::new();
        face_cascade
            .detect_multi_scale(
                &gray_frame,
                &mut faces,
                1.05,  // Scale factor
                6,     // Min neighbors (increase for fewer false positives)
                0,       // Flags
                Size::new(50, 50), // Min size
                Size::new(0, 0),   // Max size
            )
            .unwrap();

        // Draw rectangles around detected faces
        for face in faces {
            imgproc::rectangle(
                &mut frame,
                face,
                opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Color: Green
                2,                                               // Thickness
                imgproc::LINE_8,
                0,
            )
            .unwrap();
        }

        // Display the resulting frame
        highgui::imshow("Face Detection", &frame).unwrap();

        // Break the loop on pressing 'q'
        if highgui::wait_key(10).unwrap() == 113 {
            break;
        }
    }

    highgui::destroy_all_windows().unwrap();
}
