fetch('http://localhost:5000/json')
  .then(response => {
    return response.json()
  })
  .then(data => {
    // Work with JSON data here
    var main_screenshot = data.output.image
    var black_and_white_affordance_images = data.output.tag_images

    var tiles = data.output.tiles
    console.log(data.output.tiles.tile_0)

    for (var tile_name in tiles){
      var image_data = tiles[tile_name].tile_data
      var locations = tiles[tile_name]['locations']
      console.log(image_data)
      console.log(locations)
    }
  })
  .catch(err => {
    console.log('ERROR')
    // Do something for an error here
  })
