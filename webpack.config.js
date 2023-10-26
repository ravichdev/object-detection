const path = require('path');

module.exports = {
    mode: 'development',
  devServer: {
    static: {
      directory: path.join(__dirname, 'public'),
    },
    compress: true,
    port: 9000,
  },
  entry: {
    'object-detection': './src/object-detection.js',
    'person-detection': './src/person-detection.js',
  },
  output: {
    filename: '[name].js',
    path: path.resolve(__dirname, 'public/dist'),
  },
};