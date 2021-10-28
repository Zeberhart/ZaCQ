function getPosition(string, subString, index) {
  return string.split(subString, index).join(subString).length;
}