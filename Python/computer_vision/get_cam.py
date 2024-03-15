# %%
import cv2

# %%
class Stream:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_POS_MSEC, 64_000)
        cap.open(0)
        self.capture = cap

    def update(self):
        """
        Update the captured frame and return the resized frame.

        Parameters:
            self (object): The object instance
        Returns:
            numpy.ndarray: The resized frame
        """
        status, frame = self.capture.read()
        if status:
            resized_frame = cv2.resize(
                frame, (self.width, self.height), fx=0, fy=0,
                interpolation=cv2.INTER_CUBIC)
            return resized_frame
        else:
            raise Exception("Could not read frame")

# %%
running_cam = Stream(width=1024, height=512)

# %%
while True:
    try:
        current_frame = running_cam.update()
        win_title = "Frame"
        cv2.imshow(win_title, current_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(win_title,
                                                    cv2.WND_PROP_VISIBLE) < 1:
            break

    except Exception as e:
        print(f"Error: {str(e)}")
        break

# %% 
# Release the video capture
running_cam.capture.release()
cv2.destroyAllWindows()
