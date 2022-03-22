import cv2
import numpy as np


def drawline(point1, point2, image):
  """
  function to draw line

  Args:
      point1 (tuple): point 1 of the line
      point2 (tuple): point 2 of the line
      image (np.Array): numpy array of image pixel values
  """
  slope = ((point2[1]-point1[1])/(point2[0]-point1[0]))
  y_intercept = point1[1] - (slope*point1[0])

  min_x_point = min(point2[0], point1[0])
  max_x_point = max(point2[0], point1[0])
  x_pts = np.linspace(min_x_point, max_x_point, 10000)

  for x in x_pts:
    y = slope*x+y_intercept
    image[int(y), int(x)] = 1

def draw_boundary(image):
  """
  function to draw the outer box boundary for the map

  Args:
      image (np.Array): numpy array of image pixel values

  Returns:
      numpy.Array: image with boundary drawn
  """
  image_copy = image.copy()

  #  bottom boundary
  for col in range(image_copy.shape[1]):
    image_copy[0, col] = 1

  # right boundary
  for row in range(image_copy.shape[0]):
    image_copy[row, 399] = 1

  # top boundary
  for col in range(image_copy.shape[1]):
    image_copy[image_copy.shape[0]-2, col] = 1

  # left boundary
  for row in range(image_copy.shape[0]):
    image_copy[row, 0] = 1

  return image_copy

def draw_circle(center, radius, image):
  """
  function to draw circle on the image 

  Args:
      center (tuple): center coordinate of the circle
      radius (int): radius of the circle
      image (np.array): numpy array of image pixel values

  Returns:
      numpy.Array: image with circle drawn
  """
  image_copy = image.copy()

  for col  in range(image_copy.shape[1]):
      for row in range(image_copy.shape[0]):
        if(((col-center[0])**2+(row-center[1])**2) < radius**2):
              image_copy[row,col]= 1.0

  return image_copy

def half_planes(point1,point2, image, side):
  """
  function to construct half plane for a line with two points

  Args:
      point1 (tuple): point 1 of the line
      point2 (tuple): point 2 of the line
      image (np.Array): numpy array of image pixel values
      side (int): upper of lower half plane

  Returns:
      numpy.Array: image with half plane drawn
  """
  image_copy = image.copy()
  slope = (point2[0]-point1[0])/(point2[1]-point1[1]+(1e-6))
  for y  in range(1,400):
      intercept = point1[0] - slope*point1[1]
      for x in range(1,250):
          if side :
              if (y < ((slope*x)+intercept)):
                  image_copy[x,y]= 1.0
          else:
              if (y > ((slope*x)+intercept)):
                  image_copy[x,y]= 1.0

  return image_copy

def draw_hexagon(center, circumRadius, image):
  """
  function to draw hexagon on the image

  Args:
      center (tuple): center of the hexagon
      circumRadius (int): circum radius of the hexagon
      image (np.Array): numpy array of image pixel values

  Returns:
      numpy.Array: image with hexagon drawn
  """
  image_copy = image.copy()

  xpts = [center[0]+circumRadius*np.sin(angle* np.pi/180) for angle in range(180, 540, 60)]
  ypts = [center[1]+circumRadius*np.cos(angle* np.pi/180) for angle in range(180, 540, 60)]

  side_1_l = half_planes((xpts[0], ypts[0]), (xpts[1], ypts[1]), image_copy, 0)
  side_2_l = half_planes((xpts[1], ypts[1]), (xpts[2], ypts[2]), image_copy, 0)
  side_3_l = half_planes((xpts[2], ypts[2]), (xpts[3], ypts[3]), image_copy, 0)
  side_4_l = half_planes((xpts[3], ypts[3]), (xpts[4], ypts[4]), image_copy, 1)
  side_5_l = half_planes((xpts[4], ypts[4]), (xpts[5], ypts[5]), image_copy, 1)
  side_6_l = half_planes((xpts[5], ypts[5]), (xpts[0], ypts[0]), image_copy, 1)

  image_copy = cv2.bitwise_and(side_1_l, side_2_l)
  image_copy = cv2.bitwise_and(image_copy, side_3_l)
  image_copy = cv2.bitwise_and(image_copy, side_4_l)
  image_copy = cv2.bitwise_and(image_copy, side_5_l)
  image_copy = cv2.bitwise_and(image_copy, side_6_l)

  return image_copy

def draw_polygon(xpts, ypts, image):
  """
  function to draw a n sided irregular polygon

  Args:
      xpts (list): list of all x points
      ypts (list): list of all y points
      image (np.Array): numpy array of image pixel values

  Returns:
      numpy.Array: image with polygon drawn
  """
  image_copy = image.copy()

  side_1_l = half_planes((xpts[0], ypts[0]), (xpts[1], ypts[1]), image, 0)
  side_2_u = half_planes((xpts[1], ypts[1]), (xpts[2], ypts[2]), image, 1)
  side_3_l = half_planes((xpts[2], ypts[2]), (xpts[3], ypts[3]), image, 1)
  side_4_u = half_planes((xpts[3], ypts[3]), (xpts[0], ypts[0]), image, 0)

  side_1_2 = cv2.bitwise_and(side_1_l, side_2_u)
  side_1_2_3 = cv2.bitwise_and(side_1_2, side_4_u)
  side_3_4 = cv2.bitwise_and(side_4_u, side_3_l)

  side_2_l = half_planes((xpts[1], ypts[1]), (xpts[2], ypts[2]), image, 0)
  side_3_4_2 = cv2.bitwise_and(side_3_4, side_2_l)

  result = cv2.bitwise_or(side_1_2_3, side_3_4_2)

  return result

def colorize_image(image, color):
  """
  function to colorize the image

  Args:
      image (numpy.Array): numpy array of image pixel values
      color (tuple): (B, G, R) - color value

  Returns:
      numpy.Array: colourized image
  """
  image_copy = image.copy()
  color_img = np.full((image_copy.shape[0], image_copy.shape[1], 3), [241, 239, 236], dtype=np.uint8)
  for col  in range(image_copy.shape[1]):
      for row in range(image_copy.shape[0]):
        if image_copy[row,col]== 1.0:
            color_img[row,col] = color
  
  return color_img

def overlay_boundary(map_image, boundary_image, boundary_color):
  """
  function to overlay obstacle boundary on map image

  Args:
      map_image (numpy.Array): numpy array of map image pixel values
      boundary_image (numpy.Array): numpy array of boundary image pixel values
      boundary_color (tuple): (B, G, R) - boundary color value

  Returns:
      numpy.Array: colourized image along with obstacle boundary
  """

  map_image_copy = map_image.copy()
  for col  in range(boundary_image.shape[1]):
      for row in range(boundary_image.shape[0]):
        if boundary_image[row,col]== 1.0:
            map_image_copy[row,col] = boundary_color

  return map_image_copy

              
