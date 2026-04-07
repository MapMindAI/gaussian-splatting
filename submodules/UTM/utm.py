import ctypes

# 加载共享库
lib = ctypes.CDLL('submodules/UTM/libutm.so')

# 定义函数的参数类型和返回类型
lib.LatLonToUTMXY.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
lib.LatLonToUTMXY.restype = None

def LatLonToUTMXY(lat, lon, zone):
    x = ctypes.c_double()
    y = ctypes.c_double()
    lib.LatLonToUTMXY(lat, lon, zone, ctypes.byref(x), ctypes.byref(y))
    return x.value, y.value


# 测试
if __name__ == "__main__":
    lat = 34.0522
    lon = -118.2437
    zone = 11
    x, y = LatLonToUTMXY(lat, lon, zone)
    print(f"Latitude: {lat}, Longitude: {lon}")
    print(f"UTM Zone: {zone}, X: {x}, Y: {y}")
