// Regular map
function regular_map() {
    //var var_location = new google.maps.LatLng(32.5742, 73.4828);
    var var_location = new google.maps.LatLng(32.5742, 73.4828);

    var var_mapoptions = {
        center: var_location,
        zoom: 14
    };

    var var_map = new google.maps.Map(document.getElementById("map-container-7"),
        var_mapoptions);

    var var_marker = new google.maps.Marker({
        position: var_location,
        map: var_map,
        title: "MandiBahauddin"
    });
}

google.maps.event.addDomListener(window, 'load', regular_map);