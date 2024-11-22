use mpi::point_to_point as p2p;
use mpi::topology::Communicator;
use mpi::traits::*;
use opencv::{
    core::{self, Rect},
    highgui, imgproc, objdetect, prelude::*,
};

fn main() -> opencv::Result<()> {
    // Initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    // Path to Haar cascade file
    let cascade_path = "haarcascade_profileface.xml";
    let face_cascade = objdetect::CascadeClassifier::new(cascade_path)
        .expect("Failed to load Haar cascade file");

    if rank == 0 {
        // Rank 0: Master process
        let stream_url = "http://192.168.178.21:4747/video";
        let mut video_capture = videoio::VideoCapture::from_file(stream_url, videoio::CAP_FFMPEG)
            .expect("Failed to open the camera stream");

        if !video_capture.is_opened()? {
            panic!("Error: Could not open the camera stream.");
        }

        highgui::named_window("Profile Face Detection", highgui::WINDOW_AUTOSIZE)?;

        loop {
            let mut frame = Mat::default();
            if !video_capture.read(&mut frame)? || frame.empty() {
                break;
            }

            // Serialize the frame (convert it to bytes)
            let frame_data = frame.data_bytes()?.to_vec();

            // Distribute frames to workers
            for i in 1..size {
                world.process_at_rank(i).send(&frame_data);
            }

            // Collect results from workers
            for _ in 1..size {
                let mut status = p2p::Status::undefined();
                let recv_frame: Vec<u8> = world.process_at_rank(1).receive_with_tag(0, &mut status);
                let processed_frame = Mat::from_slice(&recv_frame)?;
                highgui::imshow("Profile Face Detection", &processed_frame)?;
            }

            if highgui::wait_key(1)? == 'q' as i32 {
                break;
            }
        }
    } else {
        // Worker processes
        let mut face_cascade = objdetect::CascadeClassifier::new(cascade_path)
            .expect("Failed to load Haar cascade file");

        loop {
            // Receive the frame from master
            let mut status = p2p::Status::undefined();
            let frame_data: Vec<u8> = world.process_at_rank(0).receive_with_tag(0, &mut status);

            // Deserialize the frame
            let mut frame = Mat::from_slice(&frame_data)?;

            // Convert frame to grayscale
            let mut gray_frame = Mat::default();
            imgproc::cvt_color(&frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0)?;

            // Detect profile faces
            let mut faces = core::Vector::<Rect>::new();
            face_cascade.detect_multi_scale(
                &gray_frame,
                &mut faces,
                1.1,
                5,
                0,
                core::Size::new(30, 30),
                core::Size::new(300, 300),
            )?;

            // Draw rectangles around detected profile faces
            for face in faces {
                imgproc::rectangle(
                    &mut frame,
                    face,
                    core::Scalar::new(255.0, 0.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0,
                )?;
            }

            // Serialize the processed frame and send it back
            let processed_data = frame.data_bytes()?.to_vec();
            world.process_at_rank(0).send(&processed_data);
        }
    }

    Ok(())
}
