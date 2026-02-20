import struct

class JanusVision:
    def __init__(self, raw_data, w, h):
        self.data = raw_data
        self.w = w
        self.h = h

    def find_color(self, target_bgr, tolerance=10):
        for y in range(0, self.h, 2): # Stepped for speed
            for x in range(0, self.w, 2):
                idx = (y * self.w + x) * 4
                b, g, r = self.data[idx:idx+3]
                if abs(b-target_bgr[0]) < tolerance and \
                   abs(g-target_bgr[1]) < tolerance and \
                   abs(r-target_bgr[2]) < tolerance:
                    return x, y
        return None

    def find_template(self, needle_data, nw, nh):
        # Optimization: Scan 1D byte strings for the first row of needle
        first_row = needle_data[:nw*4]
        start = 0
        while True:
            idx = self.data.find(first_row, start)
            if idx == -1: break
            x = (idx // 4) % self.w
            y = (idx // 4) // self.w
            # Verify remaining rows
            match = True
            for ny in range(1, nh):
                h_idx = ((y + ny) * self.w + x) * 4
                n_idx = (ny * nw) * 4
                if self.data[h_idx:h_idx+nw*4] != needle_data[n_idx:n_idx+nw*4]:
                    match = False
                    break
            if match: return x, y
            start = idx + 4
        return None
