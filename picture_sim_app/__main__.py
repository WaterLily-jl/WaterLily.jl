import cv2
import subprocess
import sys


def capture_image(image_name: str = "input.png") -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found or cannot be opened.")

    print("Press [space] to capture image, or [ESC] to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == 32:  # Spacebar
            path = f"input/{image_name}"
            cv2.imwrite(path, frame)
            print(f"Image saved to {path}")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Read the saved image for cropping
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to load image {path} for cropping.")

    # Let the user select ROI with mouse
    print("Select ROI (drag mouse to crop). Press ENTER or SPACE to confirm, or 'c' to cancel.")
    roi = cv2.selectROI("Crop Image", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w > 0 and h > 0:
        # Crop and overwrite the image
        cropped_img = img[int(y):int(y+h), int(x):int(x+w)]
        cv2.imwrite(path, cropped_img)
        print(f"Cropped image saved to {path}")
    else:
        print("No crop selected, original image kept.")


def main() -> None:
    capture_image()

    # Paths
    input_path = "./input/input.png"
    # input_path = "input/airfoil_30_deg.png"
    output_gif = "output/output.gif"
    julia_script = "../test/TestPixelCamSim.jl"

    # # Call Julia
    # result = subprocess.run(
    #     ["julia", julia_script, input_path, output_gif],
    #     capture_output=True,
    #     text=True
    # )

    # # Print logs or errors
    # print("Julia stdout:\n", result.stdout)
    # print("Julia stderr:\n", result.stderr)

    # # Run Julia with live output
    # cmd = ["julia", julia_script, input_path, output_gif]
    # print(f"Starting Julia: {' '.join(cmd)}\n")

    # with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
    #     for line in proc.stdout:
    #         print("[julia]", line, end='')

    # # Optional: check if Julia exited cleanly
    # if proc.returncode != 0:
    #     print(f"\nJulia process exited with code {proc.returncode}")

    cmd = ["julia", julia_script, input_path, output_gif]
    print(f"Starting Julia: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"\nJulia process exited with code {result.returncode}")


if __name__ == "__main__":
    main()
