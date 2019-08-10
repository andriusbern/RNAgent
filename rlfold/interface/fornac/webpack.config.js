var path = require('path');
var webpack = require('webpack');
console.log('process.env.NODE_PATH:', process.env.NODE_PATH);

module.exports = {
  context: __dirname + '/app',
  entry: {fornac: './scripts/fornac.js',
      rnaplot: ['./scripts/rnaplot.js'],
      rnatreemap: './scripts/rnatreemap.js'},
  output: {
    path: __dirname + '/build',
    filename: '[name].js',
    libraryTarget: 'umd',
    library: '[name]'
  },
  module: {
    loaders: [
      { 
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'babel-loader',
        query: {
          presets: ['es2015']
        }
      }, {
        test: /\.css$/,
        loader: 'style!css'
      }
    ],
    resolve: {
        fallback: process.env.NODE_PATH,
        modulesDirectories: [ process.env.NODE_PATH || "node_modules" ],
      extensions: ['.js', '.jsx']
    }
  }
};
