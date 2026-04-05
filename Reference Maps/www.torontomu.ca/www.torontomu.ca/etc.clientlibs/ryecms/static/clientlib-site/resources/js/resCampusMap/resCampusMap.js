const allOverlays = [];
let RyeUpperLeftCorner;
let RyeLowerRightCorner;
let campusBoundingBox;
let centerLat, centerLng;
let campusCenter;
const nodes = [];
let services;
const entranceNodes = [];
const points = [];
let marker;
const overlayMarkers = [];
let markerInfoWindow;
const buildings = [];
const labels = [];
let map;
let myOptions;
let layer;
let newlatlng;
let latlng;
let image;
let shadow;
const defaultIconFile = 'https://maps.google.com/mapfiles/ms/icons/red-dot.png';
const iconAccessibleRed = '/content/dam/maps/assets/Images/icons/red-dot-accessible.png';
const iconAccessibleGreen = '/content/dam/maps/assets/Images/icons/green-dot-accessible.png';
const defaultGreenIconFile = 'https://maps.google.com/mapfiles/ms/icons/green-dot.png';
const endRef = {};
const suggestedLocationMarkers = [];
let infowindow;
const overlayTypes = [];
const ayOverlayMarkers = [];
const mc = [];
const sidebar_html = [];

function getRawInfo() {
    let promises = [];

    const requests = [{
            method: "getBuildingData",
            successCallback: saveBuildings
        },
        {
            method: "getRouteNodes",
            successCallback: saveRouteNodes
        },
        {
            method: "getOverlays",
            successCallback: saveOverlays
        }
    ];

    requests.forEach(function(request) {
        const promise = new Promise(function(resolve, reject) {
            jQuery.ajax({
                type: "GET",
                url: "https://m.torontomu.ca/core_apps/map/components/map_funcs.cfc",
                data: {
                    method: request.method
                },
                dataType: "json",
                success: function(data) {
                    resolve(data);
                },
                error: function(xhr, status, error) {
                    reject(error);
                }
            });
        });

        promises.push(promise);
    });

    Promise.all(promises)
        .then(function(results) {
            results.forEach(function(result, index) {
                requests[index].successCallback(result);
            });
            parseParams();
        })
        .catch(function(error) {
            console.error("Error occurred:", error);
        });
}

function init() {
    RyeUpperLeftCorner = new google.maps.LatLng(43.662268, -79.385236);
    RyeLowerRightCorner = new google.maps.LatLng(43.655376, -79.374051);
    campusBoundingBox = new google.maps.Polyline();
    centerLat = ((RyeUpperLeftCorner.lat() - RyeLowerRightCorner.lat()) / 2) + RyeLowerRightCorner.lat();
    centerLng = ((RyeLowerRightCorner.lng() - RyeUpperLeftCorner.lng()) / 2) + RyeUpperLeftCorner.lng();
    campusCenter = new google.maps.LatLng(centerLat, centerLng);
    marker = new google.maps.Marker();
    markerInfoWindow = new google.maps.InfoWindow();
    latlng = new google.maps.LatLng(43.657929, -79.38122034072876);
    image = new google.maps.MarkerImage("//www.ryerson.ca/maps/assets/Images/icons/youarehere.png", new google.maps.Size(38.0, 40.0), new google.maps.Point(0, 0), new google.maps.Point(19.0, 20.0));
    shadow = new google.maps.MarkerImage("//www.ryerson.ca/maps/assets/Images/icons/shadow-youarehere.png", new google.maps.Size(59.0, 40.0), new google.maps.Point(0, 0), new google.maps.Point(19.0, 20.0));
    //center of the campus
    myOptions = {
        zoom: 16,
        center: latlng,
        mapTypeId: google.maps.MapTypeId.ROADMAP,
        styles: [{
            featureType: "poi",
            elementType: "labels",
            stylers: [{
                visibility: "off"
            }]
        }, {
            featureType: "landscape.man_made",
            elementType: "labels",
            stylers: [{
                visibility: "off"
            }]
        }],
        disableDoubleClickZoom: true,
        streetViewControl: false
    };
    map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);
    getRawInfo();
    google.maps.event.addListener(map, 'zoom_changed', zoomToggle);
    initCustom();
}

function parseClickHandler() {
    return function() {
        const buildingCode = this.code;
        let buildingIndex;
        for (let i in buildings) {
            if (buildings[i].outline.code === buildingCode) {
                buildingIndex = i;
                break;
            }
        }
        const point = makePoint((buildings[buildingIndex].outline.name + ", " + buildings[buildingIndex].outline.code), buildings[buildingIndex].outline.center, "IH", "");
        points.length = 0;
        points.push(point);
        processRequest();
    };
}

function parseParams() {
    if (getParameter('building') !== 'null') {
        for (let i in buildings) {
            if (getParameter('building').toUpperCase().indexOf(buildings[i].outline.code) !== -1) {
                map.setCenter(buildings[i].outline.center);
                break;
            }
        }
    }
}

function saveBuildings(result) {
    let CoOrds = [];
    let allCoOrds = [];
    let googCoOrds = [];
    //split the polygon into an array of coordinates
    //create an array of arrays to store the co-ords
    for (let x in result) {
        if (result[x].hasOwnProperty('POLYGON')) {
            CoOrds = result[x].POLYGON.split(",\r");
            allCoOrds[x] = CoOrds;
        }
    }
    for (let i in allCoOrds) {
        if (allCoOrds.hasOwnProperty(i)) {
            googCoOrds[i] = [];
            //creates an array of co-ordinates for each building to draw polygon
            for (let j in allCoOrds[i]) {
                if (allCoOrds[i].hasOwnProperty(j)) {
                    const lat = allCoOrds[i][j].split(",")[0];
                    const lon = allCoOrds[i][j].split(",")[1];
                    newlatlng = new google.maps.LatLng(lat, lon);
                    googCoOrds[i] = googCoOrds[i].concat(newlatlng);
                }
            }
        }
    }
    //create an array of buildings that holds the polygon, name, center point and id of the building
    for (let i in googCoOrds) {
        if (googCoOrds.hasOwnProperty(i)) {
            buildings[i] = {};
            const fillColor = "#5289bf";
            const strokeColor = "#ffffff";
            buildings[i].outline = new google.maps.Polygon({
                paths: googCoOrds[i],
                strokeColor: strokeColor,
                strokeOpacity: 0.8,
                strokeWeight: 1,
                fillColor: fillColor,
                fillOpacity: 1.0
            });
            buildings[i].outline.id = result[i].ID;
            buildings[i].outline.name = result[i].NAME;
            buildings[i].outline.code = result[i].CODE;
            buildings[i].outline.center = result[i].CENTER;
            buildings[i].outline.address = result[i].ADDRESS;
            buildings[i].outline.details = result[i].DETAILS;
            buildings[i].outline.attributes = {};
            buildings[i].outline.attributes.wheelchair = result[i].WHEELCHAIR;
            buildings[i].outline.attributes.elevator = result[i].ELEVATOR;
            buildings[i].outline.video_ref = result[i].VIDEO_REF;
            buildings[i].outline.hours = result[i].HOURS;
            buildings[i].outline.accessibility = result[i].ACCESSIBILITY;
            buildings[i].outline.defaultEntrance = result[i].DEFAULT_ENTRANCE;
            buildings[i].outline.setMap(map);
            const f = parseClickHandler();
            google.maps.event.addListener(buildings[i].outline, 'click', f);
        }
    }
    //assigns nodes to the entrances of buildings
    for (let i in buildings) {
        if (buildings[i].hasOwnProperty('outline')) {
            let index = 0;
            buildings[i].outline.entrances = [];
            for (let x in entranceNodes) {
                //check for a substring match between buildings code (ie KHW) and the entrance nodes name
                if (entranceNodes[x].name.indexOf(buildings[i].outline.code) !== -1) {
                    buildings[i].outline.entrances[index] = entranceNodes[x].name;
                    index += 1;
                }
            }
            const lat = buildings[i].outline.center.split(",")[0];
            const lon = buildings[i].outline.center.split(",")[1];
            const labelCoOrd = new google.maps.LatLng(lat, lon);
            buildings[i].outline.center = new google.maps.LatLng(lat, lon);
            buildings[i].outline.labels = {};
            const codeInfo = '<div> <div class="building-marker">' + buildings[i].outline.code + '<\/div><\/div>';
            buildings[i].outline.labels.code = (new RichMarker({
                map: map,
                position: labelCoOrd,
                flat: true,
                draggable: false,
                anchor: RichMarkerPosition.MIDDLE,
                content: codeInfo
            }));
            buildings[i].outline.labels.code.setMap(map);
            buildings[i].outline.labels.code.setVisible(true);
            if (map.getZoom() < 16) {
                buildings[i].outline.labels.code.setVisible(false);
            }
        }
    }
    result = null;
}

function saveRouteNodes(nodesCF) {
    let nodesIndex = 0;
    let entranceIndex = 0;
    for (let x in nodesCF) {
        const lat = nodesCF[x].POSITION.split(",")[0];
        const lon = nodesCF[x].POSITION.split(",")[1];
        newlatlng = new google.maps.LatLng(lat, lon);
        let hasEntrance = false;
        let isAccessible = false;
        if ((typeof nodesCF[x].ATTRIBUTE) != 'undefined') {
            for (let y in nodesCF[x].ATTRIBUTE) {
                if (nodesCF[x].ATTRIBUTE[y].split(":")[0] === "Entrance") {
                    hasEntrance = true;
                }
                if (nodesCF[x].ATTRIBUTE[y].split(":")[0] === "Accessibility") {
                    isAccessible = true;
                }
            }
        }
        if (hasEntrance) {
            entranceNodes[entranceIndex] = {}
            entranceNodes[entranceIndex].name = nodesCF[x].POINT;
            entranceNodes[entranceIndex].position = newlatlng;
            entranceNodes[entranceIndex].neighbors = nodesCF[x].NEIGHBORS.split(",");
            entranceNodes[entranceIndex].attributes = {};
            entranceNodes[entranceIndex].attributes.accessible = isAccessible;
            entranceIndex += 1;
        }
        nodes[nodesIndex] = {};
        nodes[nodesIndex].attributes = {};
        nodes[nodesIndex].id = nodesCF[x].ID;
        nodes[nodesIndex].name = nodesCF[x].POINT;
        nodes[nodesIndex].street = nodesCF[x].STREET;
        nodes[nodesIndex].position = newlatlng;
        nodes[nodesIndex].neighbors = nodesCF[x].NEIGHBORS.split(",");
        nodes[nodesIndex].attributes.accessible = isAccessible;
        nodes[nodesIndex].attributes.entrance = hasEntrance;
        nodesIndex += 1;
    }
}

function saveOverlays(servicesCF) {
    for (let i in servicesCF) {
        const lat = servicesCF[i].SITE.split(",")[0];
        const lon = servicesCF[i].SITE.split(",")[1];
        const latlng = new google.maps.LatLng(lat, lon);
        allOverlays[i] = {};
        allOverlays[i].id = servicesCF[i].ID;
        if (servicesCF[i].TYPE === 'Parking') servicesCF[i].TYPE = 'Car Parking';
        allOverlays[i].type = servicesCF[i].TYPE;
        allOverlays[i].site = latlng;
        allOverlays[i].description = servicesCF[i].DESCRIPTION;
        allOverlays[i].icon = servicesCF[i].ICON;
        allOverlays[i].moreInfo = servicesCF[i].MORE_INFO;
    }
    let overlayServices = [];
    for (let i in allOverlays) {
        if (jQuery.inArray(allOverlays[i].type, overlayServices) === -1) {
            overlayServices.push(allOverlays[i].type);
        }
    }
}

function zoomToggle() {
    const zoom = map.getZoom();
    if (zoom > 15) {
        for (let i in buildings) {
            buildings[i].outline.labels.code.setVisible(true);
        }
    } else {
        for (let i in buildings) {
            buildings[i].outline.labels.code.setVisible(false);
        }
    }
}

function getParameter(paramName) {
    const searchString = window.location.search.substring(1),
        params = searchString.split("&");
    let val;
    for (let i = 0; i < params.length; i++) {
        val = params[i].split("=");
        if (val[0] === paramName) {
            return val[1];
        }
    }
    return 'null';
}

function initCustom() {
    jQuery(".navbar-header .navbar-toggle").hide();
    map.setCenter(new google.maps.LatLng(43.6588682, -79.38222034072876));
    google.maps.event.addListener(map, 'zoom_changed', infowindowclose);
    google.maps.event.addListener(map, 'drag', infowindowclose); // needed to avoid "floating direction textbox"
    waitForCompletion();
}

function makePoint(label, location, type, reference) {
    const point = {};
    point.label = label;
    point.location = location;
    point.type = type;
    if (type === "G") {
        point.placeId = reference;
    }
    return point;
}

function processRequest() {
    for (let x in suggestedLocationMarkers) {
        suggestedLocationMarkers[x].marker.setMap(null);
    }
    suggestedLocationMarkers.length = 0;
    marker.setMap(null);
    if (points.length === 1) {
        let buildingCode = points[0].label.split(", ")[0];
        if (points[0].type !== "KW") {
            for (let x in buildings) {
                if (buildings[x].outline.name === buildingCode) {
                    buildingCode = buildings[x].outline.code;
                    showBuildingInfo(x);
                    break;
                }
            }
        }
        const markerArray = [];
        if (points[0].type === "IH") {
            //ONE building on campus returns array of entrances
            //can just match node names
            for (let x in nodes) {
                if (nodes[x].name.match(new RegExp(buildingCode, 'gi')) != null) {
                    markerArray.push(nodes[x]);
                }
            }
            for (let x in markerArray) {
                suggestedLocationMarkers[x] = {};
                suggestedLocationMarkers[x].marker = new google.maps.Marker({
                    position: markerArray[x].position,
                    map: map,
                    title: markerArray[x].name
                });
                suggestedLocationMarkers[x].marker.accessible = markerArray[x].attributes.accessible === true;
                suggestedLocationMarkers[x].marker.setMap(map);
                suggestedLocationMarkers[x].marker.setVisible(true);
                if (suggestedLocationMarkers[x].marker.accessible === true) {
                    suggestedLocationMarkers[x].marker.setIcon(iconAccessibleRed);
                } else {
                    suggestedLocationMarkers[x].marker.setIcon(defaultIconFile);
                }
            }
            if (suggestedLocationMarkers.length) {
                if (suggestedLocationMarkers[0].marker.accessible === true) {
                    suggestedLocationMarkers[0].marker.setIcon(iconAccessibleGreen);
                } else {
                    suggestedLocationMarkers[0].marker.setIcon(defaultGreenIconFile);
                }
            }
        } else if (points[0].type === "G") {
            //ONE result from google maps
            suggestedLocationMarkers[0] = {};
            suggestedLocationMarkers[0].marker = new google.maps.Marker({
                position: points[0].location,
                map: map,
                title: points[0].label
            });
            suggestedLocationMarkers[0].marker.setMap(map);
            suggestedLocationMarkers[0].marker.setVisible(true);
            suggestedLocationMarkers[0].marker.setIcon(defaultGreenIconFile);
            endRef.placeId = points[0].placeId;
        } else if (points[0].type === "KW") {
            suggestedLocationMarkers[0] = {};
            suggestedLocationMarkers[0].marker = new google.maps.Marker({
                position: points[0].location,
                map: map,
                title: buildingCode
            });
            suggestedLocationMarkers[0].marker.setIcon(defaultGreenIconFile);
        }
    }
}

function aboutbuilding(id) {
    const d1 = document.getElementById('rmap_building' + id + '_details');
    const d2 = document.getElementById('rmap_direction' + id + '_details');
    const d3 = document.getElementById('rmap_hours' + id + '_details');
    const d4 = document.getElementById('rmap_info' + id);
    const d5 = document.getElementById('rmap_dirs' + id);
    const d6 = document.getElementById('rmap_hours' + id);
    d1.style.display = "block";
    d2.style.display = "none";
    d3.style.display = "none";
    d4.className = "current";
    d5.className = "";
    d6.className = "";
}

function gethours(id) {
    const d1 = document.getElementById('rmap_building' + id + '_details');
    const d2 = document.getElementById('rmap_direction' + id + '_details');
    const d3 = document.getElementById('rmap_hours' + id + '_details');
    const d4 = document.getElementById('rmap_info' + id);
    const d5 = document.getElementById('rmap_dirs' + id);
    const d6 = document.getElementById('rmap_hours' + id);
    d1.style.display = "none";
    d2.style.display = "none";
    d3.style.display = "block";
    d4.className = "";
    d5.className = "";
    d6.className = "current";
}

function showdirections(id) {
    const d1 = document.getElementById('rmap_building' + id + '_details');
    const d2 = document.getElementById('rmap_direction' + id + '_details');
    const d3 = document.getElementById('rmap_hours' + id + '_details');
    const d4 = document.getElementById('rmap_info' + id);
    const d5 = document.getElementById('rmap_dirs' + id);
    const d6 = document.getElementById('rmap_hours' + id);
    d1.style.display = "none";
    d2.style.display = "block";
    d3.style.display = "none";
    d4.className = "";
    d5.className = "current";
    d6.className = "";
}

function infowindowclose() {
    if (infowindow) infowindow.close();
}

function boxclick(el, cat) {
    if (el.checked) {
        if (!ayOverlayMarkers[cat] || ayOverlayMarkers[cat].length === 0) {
            let legendcatset = false;
            let mcOptions;
            for (let i in allOverlays) {
                if (allOverlays[i].type === cat) {
                    if (!legendcatset) {
                        let typeFileName;
                        if (allOverlays[i].type.indexOf('OneCard') !== -1 || allOverlays[i].type.indexOf('One Card') !== -1) typeFileName = 'One Card';
                        else typeFileName = allOverlays[i].type;
                        mcOptions = {
                            gridSize: 50,
                            maxZoom: 17,
                            ignoreHidden: true,
                            styles: [{
                                url: '/content/dam/maps/icons2/' + typeFileName + '-cluster.png',
                                width: 32,
                                height: 39,
                                anchorText: [-12, 8],
                                textColor: '#000000',
                                textSize: 10
                            }]
                        };
                        ayOverlayMarkers[cat] = [];
                        legendcatset = true;
                    }
                    let typeFileName;
                    if (allOverlays[i].type.indexOf('OneCard') !== -1 || allOverlays[i].type.indexOf('One Card') !== -1) typeFileName = 'One Card';
                    else typeFileName = allOverlays[i].type;
                    const marker = new google.maps.Marker({
                        position: new google.maps.LatLng(allOverlays[i].site.lat(), allOverlays[i].site.lng()),
                        map: map,
                        icon: '/content/dam/maps/icons2/' + typeFileName + '-marker.png',
                        description: allOverlays[i].description,
                        moreInfo: allOverlays[i].moreInfo.replace("<a href=", '<a target="_blank" href=')

                    });
                    ayOverlayMarkers[cat].push(marker);
                    google.maps.event.addListener(marker, "click", function() {
                        infowindowclose();
                        infowindow = new google.maps.InfoWindow({
                            content: '<div class="rmap_infocontent_marker"><h1 id="firstHeading" class="firstHeading">' + this.description + '</h1>' + this.moreInfo + '</div>'
                        });
                        google.maps.event.addListener(infowindow, 'closeclick', infowindowclose);
                        infowindow.open(map, this);
                    });
                }
            }
            mc[cat] = new MarkerClusterer(map, ayOverlayMarkers[cat], mcOptions);
        } else {
            showOverlay(cat);
        }
    } else {
        hideOverlay(cat);
    }
}

function showOverlay(cat) {
    for (let i in ayOverlayMarkers[cat]) {
        ayOverlayMarkers[cat][i].setVisible(true);
    }
    // == check the checkbox ==
    document.getElementById(cat + "box").checked = true;
    // == close the info window, in case its open on a marker that we just hid
    infowindowclose();
    // recount clusters (if any)
    mc[cat].repaint();
}

function hideOverlay(cat) {
    for (let i in ayOverlayMarkers[cat]) {
        ayOverlayMarkers[cat][i].setVisible(false);
    }
    ayOverlayMarkers[cat] = [];
    // == clear the checkbox ==
    document.getElementById(cat + "box").checked = false;
    // recount clusters (if any)
    mc[cat].repaint();
    mc[cat] = null;
}

function customSortBuildings(a, b) {
    if (a.outline.name < b.outline.name)
        return -1;
    if (a.outline.name > b.outline.name)
        return 1;
    return 0;
}

function generateLegend() {
    for (let i in allOverlays) {
        if (jQuery.inArray(allOverlays[i].type, overlayTypes) === -1) {
            overlayTypes.push(allOverlays[i].type);
        }
    }
    overlayTypes.sort();
    let legend = '';
    for (let i in overlayTypes) {
        let typeFileName;
        if (overlayTypes[i].indexOf('OneCard') !== -1 || overlayTypes[i].indexOf('One Card') !== -1) typeFileName = 'One Card';
        else typeFileName = overlayTypes[i];
        legend += '<li><label for="' + overlayTypes[i] + 'box"><img width="32" height="39" src="/content/dam/maps/icons2/' + typeFileName + '-icon.png" alt="' + overlayTypes[i] + '">&nbsp;' + sentenceCase(overlayTypes[i]) + '</label><input type="checkbox" id="' + overlayTypes[i] + 'box" onclick="boxclick(this,\'' + overlayTypes[i] + '\')"></li>';
    }
    jQuery("#legendlist").html(legend);
    buildings.sort(customSortBuildings);
    for (let x in buildings) {
        let underConstruction = '';
        //if (buildings[x].outline.code == 'DCC') underConstruction = ' <span style="color: red;">(under construction)</span>';
        sidebar_html.push('<div id="sidebar_b_' + x + '"><a href="javascript:;" onclick="showBuildingInfo(\'' + x + '\')"><span class="bcode">' + buildings[x].outline.name + '</span> (' + buildings[x].outline.code + ')' + underConstruction + '</a></div>');
    }
    const splithtml = split(sidebar_html, 2);
    jQuery("#sidebar0").html(splithtml[0].join(""));
    jQuery("#sidebar1").html(splithtml[1].join(""));
    const buildingparam = window.location.search.substring(10, window.location.search.length).toUpperCase();
    if (typeof buildingparam !== 'undefined') {
        for (let i in buildings) {
            if (buildings[i].outline.code === buildingparam) {
                showBuildingInfo(i);
                break;
            }
        }
    }
}

function sentenceCase(str) {
    if (str.indexOf('OneCard') !== -1) return str;
    str = str.toLowerCase();
    str = str.charAt(0).toUpperCase() + str.slice(1);
    return str;
}

function split(a, n) {
    const len = a.length,
        out = [];
    let i = 0;
    while (i < len) {
        const size = Math.ceil((len - i) / n--);
        out.push(a.slice(i, i += size));
    }
    return out;
}

function waitForCompletion() {
    if (allOverlays.length === 0) {
        setTimeout(waitForCompletion, 100);
    } else {
        generateLegend();
    }
}

function showBuildingInfo(buildingInfoIndex) {
    let wheelchair_info = "";
    let elevator_info = "";
    if (buildings[buildingInfoIndex].outline.attributes.wheelchair === "yes") {
        wheelchair_info = '<img src="/content/dam/maps/images/wheelchair.gif" alt="Wheelchair-accessible external entrance" title="Wheelchair-accessible external entrance">&nbsp;';
    }
    if (buildings[buildingInfoIndex].outline.attributes.elevator === "yes") {
        elevator_info = '<img src="/content/dam/maps/images/elevator.gif" alt="Elevator" title="Elevator">';
    }
    let deets = buildings[buildingInfoIndex].outline.details;
    let defaultEntrance;
    if (buildings[buildingInfoIndex].outline.defaultEntrance !== "") defaultEntrance = buildings[buildingInfoIndex].outline.defaultEntrance.replace(/\s+/, '');
    else defaultEntrance = buildings[buildingInfoIndex].outline.address.replace(/\s+/g, '+') + ',+Toronto,+ON';
    let contentString = '<div class="rmap_infocontent">' +
        '<div class="content_wrapper"><img src="//www.ryerson.ca/maps/images/bldg_' + buildings[buildingInfoIndex].outline.code + '.jpg" title="' + buildings[buildingInfoIndex].outline.name + '" alt="Photo of ' + buildings[buildingInfoIndex].outline.name + '"><span class="aaa"><strong class="code">' + buildings[buildingInfoIndex].outline.code + ' ' + wheelchair_info + elevator_info + '</strong><br/><strong>' + buildings[buildingInfoIndex].outline.name + '</strong><br/>' + buildings[buildingInfoIndex].outline.address + '<br/></span></div>' +
        '<DIV class=tabs_header style="width: 335px !important;"><UL class=tabs_primary style="width: 330px !important;"><LI><A href="javascript:;" onclick="aboutbuilding(' + buildings[buildingInfoIndex].outline.id + ')" id="rmap_info' + buildings[buildingInfoIndex].outline.id + '" class="current" style="width: 100px !important;">About This Building</A></LI><LI><A href="javascript:;" onclick="gethours(' + buildings[buildingInfoIndex].outline.id + ')" id="rmap_hours' + buildings[buildingInfoIndex].outline.id + '" style="width: 100px !important;';
    if (!buildings[buildingInfoIndex].outline.accessibility) contentString += ' display: none !important;';
    contentString += '">Accessibility</A></LI><LI><A href="javascript:;" onclick="showdirections(' + buildings[buildingInfoIndex].outline.id + ')" id="rmap_dirs' + buildings[buildingInfoIndex].outline.id + '" style="width: 100px !important;">Directions</A></LI></UL></DIV><DIV class=tabs_main><DIV class=tabs_contents><span class="content_wrapper">' +
        // Building Details (default)
        '<div id="rmap_building' + buildings[buildingInfoIndex].outline.id + '_details">' + deets + '</div>' +
        // Hours tab
        '<div id="rmap_hours' + buildings[buildingInfoIndex].outline.id + '_details" style="display: none;">' + buildings[buildingInfoIndex].outline.accessibility + '</div>' +
        // Directions tab
        '<div id="rmap_direction' + buildings[buildingInfoIndex].outline.id + '_details" style="display: none;">' +
        '<ul><li>Get <a href="https://www.google.com/maps/dir//' + defaultEntrance + '/" target="_blank">directions to here with Google Maps</strong> <span class="fa fa-external-link" aria-hidden="true"></span></a><ul><li><strong>Note:</strong> If you are logged into your TMU account, please open this link using incognito mode or private browsing.</li></ul></li>' +
        '<li>Get <a href="https://maps.apple.com/?daddr=' + defaultEntrance + '&dirflg=w&t=m" target="_blank">directions to here with Apple Maps</strong> <span class="fa fa-external-link" aria-hidden="true"></span></a></li>';
    if (buildings[buildingInfoIndex].outline.defaultEntrance !== "") contentString += '<li>Get <a href="https://citymapper.com/directions?endcoord=' + defaultEntrance + '&endname=' + encodeURIComponent(buildings[buildingInfoIndex].outline.name) + '" target="_blank">directions to here with Citymapper</strong> <span class="fa fa-external-link" aria-hidden="true"></span></a></li>';
    contentString += '</ul></span></DIV></DIV></DIV>';
    infowindowclose();
    infowindow = new google.maps.InfoWindow({
        content: contentString
    });
    google.maps.event.addListener(infowindow, 'closeclick', infowindowclose);
    const b_latlng = new google.maps.LatLng(buildings[buildingInfoIndex].outline.center.lat(), buildings[buildingInfoIndex].outline.center.lng());
    const b_marker = new google.maps.Marker({
        position: b_latlng,
        map: map,
        title: buildings[buildingInfoIndex].outline.name,
        icon: '/content/dam/maps/images/transparent.gif'
    });
    if (map.getZoom() < 16) map.setZoom(16);
    infowindow.open(map, b_marker);
    const mapTop = jQuery("#map_canvas").offset().top;
    if (jQuery(document).scrollTop() > mapTop) {
        jQuery("html, body").animate({
            scrollTop: mapTop
        }, "fast");
    }
    map.panTo(b_latlng);
}