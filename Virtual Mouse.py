import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import time
import HandTracking as ht
import autopy

class HandTrackingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Hand Tracking App")

        # Configure window size and position
        self.master.geometry("500x300")
        self.master.resizable(False, False)
        self.master.configure(bg="#f0f0f0")

        # Create title label
        self.title_label = tk.Label(master, text="Hand Tracking App", font=("Helvetica", 20), bg="#f0f0f0")
        self.title_label.pack(pady=(20, 10))

        # Create start button
        self.start_button = tk.Button(master, text="Start Tracking", command=self.start_tracking, font=("Helvetica", 14), bg="#4CAF50", fg="white", padx=20, pady=10)
        self.start_button.pack(pady=20)

        # Create quit button
        self.quit_button = tk.Button(master, text="Quit", command=self.quit_app, font=("Helvetica", 14), bg="#f44336", fg="white", padx=20, pady=10)
        self.quit_button.pack(pady=10)

    def start_tracking(self):
        try:
            # Your code here
            pTime = 0
            width = 640
            height = 480
            frameR = 100
            smoothening = 8
            prev_x, prev_y = 0, 0
            curr_x, curr_y = 0, 0

            cap = cv2.VideoCapture(0)
            cap.set(3, width)
            cap.set(4, height)

            detector = ht.handDetector(maxHands=1)
            screen_width, screen_height = autopy.screen.size()

            while True:
                success, img = cap.read()
                img = detector.findHands(img)
                lmlist, bbox = detector.findPosition(img)

                if len(lmlist) != 0:
                    x1, y1 = lmlist[8][1:]
                    x2, y2 = lmlist[12][1:]

                    fingers = detector.fingersUp()
                    cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)
                    if fingers[1] == 1 and fingers[2] == 0:
                        x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
                        y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

                        curr_x = prev_x + (x3 - prev_x) / smoothening
                        curr_y = prev_y + (y3 - prev_y) / smoothening

                        autopy.mouse.move(screen_width - curr_x, curr_y)
                        cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                        prev_x, prev_y = curr_x, curr_y

                    if fingers[1] == 1 and fingers[2] == 1:
                        length, img, lineInfo = detector.findDistance(8, 12, img)

                        if length < 40:
                            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            autopy.mouse.click()

                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                cv2.imshow("Image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def quit_app(self):
        self.master.destroy()

def main():
    root = tk.Tk()
    app = HandTrackingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
